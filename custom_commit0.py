import os
import sys
import yaml
import multiprocessing
import re
import logging
from agent.run_agent import DirContext, run_eval_after_each_commit
from aider.io import InputOutput
from agent.agents import AiderAgents
from datasets import load_dataset
from git import Repo
from agent.agent_utils import (
    create_branch,
    get_message,
    get_target_edit_files,
    get_changed_files_from_commits,
    update_message_with_dependencies,
    get_lint_cmd,
    read_yaml_config,
    extract_function_stubs
)
from agent.agents import AiderAgents, AgentReturn, AiderReturn, handle_logging
from aider.coders import Coder
from aider.models import Model
import json
from agent.class_types import AgentConfig
from commit0.harness.constants import SPLIT
from commit0.harness.get_pytest_ids import main as get_tests
from commit0.harness.constants import RUN_AGENT_LOG_DIR, RepoInstance
from commit0.cli import read_commit0_config_file
from pathlib import Path
from datetime import datetime
from agent.display import TerminalDisplay

from custom_agent_utils import *

### NOTE not sure if this code is compatible with the current vesrion of commit0 or not
### VERSION OF CUSTOM_RUN_AGENT_FOR_REPO which is up to date with git, not pip (11/12)
def custom_run_agent_team_for_repo(
    repo_base_dir: str,
    agent_config: AgentConfig,
    example: RepoInstance,
    branch: str,
    update_queue: multiprocessing.Queue,
    override_previous_changes: bool = False,
    backend: str = "modal",
    log_dir: str = str(RUN_AGENT_LOG_DIR.resolve()),
    commit0_config_file: str = "",
) -> None:
    """Run Aider for a given repository."""
    # get repo info
    commit0_config = read_commit0_config_file(commit0_config_file)
    
    assert "commit0" in commit0_config["dataset_name"]
    repo_name = example["repo"].split("/")[-1]
    
    # before starting, display all information to terminal
    update_queue.put(("start_repo", (repo_name, 0)))

    repo_path = os.path.join(repo_base_dir, repo_name)
    repo_path = os.path.abspath(repo_path)

    try:
        local_repo = Repo(repo_path)
    except Exception:
        raise Exception(
            f"{repo_path} is not a git repo. Check if base_dir is correctly specified."
        )

    # NOTE initiialize custom CodingAgent
        # TODO comment on the actions of the coding agent
    coder_agent = CodingAgent(agent_config.max_iteration, agent_config.model_name)

    # Check if there are changes in the current branch
    if local_repo.is_dirty():
        # Stage all changes
        local_repo.git.add(A=True)
        # Commit changes with the message "left from last change"
        local_repo.index.commit("left from last change")

    # if branch_name is not provided, create a new branch name based on agent_config
    # if branch is None:
    #     branch = args2string(agent_config)
    create_branch(local_repo, branch, example["base_commit"])

    # in cases where the latest commit of branch is not commit 0
    # set it back to commit 0
    latest_commit = local_repo.commit(branch)
    if latest_commit.hexsha != example["base_commit"] and override_previous_changes:
        local_repo.git.reset("--hard", example["base_commit"])

    # get target files to edit and test files to run
    target_edit_files, import_dependencies = get_target_edit_files(
        local_repo,
        example["src_dir"],
        example["test"]["test_dir"],
        branch,
        example["reference_commit"],
        agent_config.use_topo_sort_dependencies,
    )

    # prepare the log dir
    experiment_log_dir = (
        Path(log_dir)
        / repo_name
        / branch
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    experiment_log_dir.mkdir(parents=True, exist_ok=True)

    """
    START revert code
    """
    # Initialize baseline commit
    baseline_commit = local_repo.head.commit.hexsha

    # don't evaluate baseline performance because it could result in no code ever getting written
    # i.e. with simpy that starts out passing a lot of test cases

    best_results = {'num_passed': 0, 'num_tests': 0}
    """
    END revert code
    """

    eval_results = {}
    # write agent_config to .agent.yaml in the log_dir for record
    agent_config_log_file = experiment_log_dir / ".agent.yaml"
    with open(agent_config_log_file, "w") as agent_config_file:
        yaml.dump(agent_config, agent_config_file)

    with DirContext(repo_path):
        if agent_config is None:
            raise ValueError("Invalid input")

        else:
            file_name = "all"
            file_log_dir = experiment_log_dir / file_name
            lint_cmd = get_lint_cmd(repo_name, agent_config.use_lint_info, commit0_config_file)
            
            # iterate through target_edit_files, impl repository files in that order
            file_impl_count = {}
            i = 0
            while i < len(target_edit_files):
                file_name = target_edit_files[i]
                file_impl_count[file_name] = file_impl_count.get(file_name, 0) + 1 # incr. # of times that file has been impl.
                file_impl = False # indicator var for if a commit implementing this file is kept
                update_queue.put(("set_current_file", (repo_name, file_name)))

                with open(experiment_log_dir / "revert_log_file.txt", "a+") as f:
                    f.write(f"IMPLEMENTING {file_name}\n")
                
                #implement_message = f"Implement the incomplete functions and classes in {file_name}. There should be no raise NotImplementedErrors left in the file."
                second_half_of_test_cmd = f"--branch {branch} --commit0-config-file {commit0_config_file} --timeout 100"
                
                # call run function for custom coder agent
                # completes the implementation, runs pytest, and uses error output to make some edits (once)
                agent_return = coder_agent.run(agent_config.user_prompt, 
                    second_half_of_test_cmd, 
                    lint_cmd, 
                    [file_name], 
                    file_log_dir, 
                    repo_name=repo_name)

                """
                START revert code
                """
                # Run tests and check performance
                current_eval_results = run_eval_after_each_commit(
                    branch, backend, commit0_config_file
                )

                # Search for the target repository line
                pattern = fr"^{repo_name},[^,]+,(\d+)/(\d+)"
                match = re.search(pattern, current_eval_results, re.MULTILINE)

                if match:
                    current_num_passed_tests = int(match.group(1))
                    total_tests = int(match.group(2))
                else:
                    raise RuntimeError(f"Searching for eval results didn't work {current_eval_results}")
                current_results = {'num_passed': current_num_passed_tests, 'num_tests': total_tests}
        
                performance_improved = check_performance_improved(
                    best_results, current_results
                )

                revert_info = ""
                if performance_improved or file_impl_count[file_name] >= 3:
                    # Keep changes
                    file_impl = True
                    baseline_commit = local_repo.head.commit.hexsha
                    if file_impl_count[file_name] >= 3:
                        revert_info += f"\nMax impl. reached (no revert), current hash {baseline_commit}"
                    else:
                        revert_info += f"\nImproved perf. (no revert), current hash {baseline_commit}"
                    
                    best_results = current_results
                else:
                    # Revert changes
                    revert_info += f"\nReverted to {baseline_commit}"
                    revert_to_commit(local_repo, baseline_commit)
                
                ## LOG REVERT UPDATES in revert_log_files
                with open(experiment_log_dir / "revert_log_file.txt", "a+") as f:
                    f.write("new results: ")
                    json.dump(current_results, f)
                    f.write("\nbest results: ")
                    json.dump(best_results, f)
                    f.write(revert_info+"\n")
                    f.write("\n\n")
                """
                END revert code
                """

                update_queue.put(
                    (
                        "update_money_display",
                        (repo_name, file_name, agent_return.last_cost),
                    )
                )

                # move on to next file only if a commit implemting this file is kept
                if file_impl:
                    i += 1
                
    if agent_config.record_test_for_each_commit:
        with open(experiment_log_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f)

    update_queue.put(("finish_repo", repo_name))


class CodingAgent(AiderAgents):
    """
    Implements files in fnames list
    
    """
    def run(
        self,
        implement_message: str,
        test_cmd_second_half: str,
        lint_cmd: str,
        fnames: list[str],
        log_dir: Path,
        repo_name: str
    ) -> AgentReturn:
        if test_cmd_second_half:
            auto_test = True
        else:
            auto_test = False
        if lint_cmd:
            auto_lint = True
        else:
            auto_lint = False
        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        input_history_file = log_dir / ".aider.input.history"
        chat_history_file = log_dir / ".aider.chat.history.md"

        # Set up logging
        log_file = log_dir / "aider.log"
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Redirect print statements to the log file
        sys.stdout = open(log_file, "a")
        sys.stderr = open(log_file, "a")

        # Configure httpx and backoff logging
        handle_logging("httpx", log_file)
        handle_logging("backoff", log_file)

        # FIND ALL TEST FILES
        test_files_str = get_tests(repo_name, verbose=0)
        test_files = sorted(list(set([i.split(":")[0] for i in test_files_str])))

        io = InputOutput(
            yes=False, ## set yes to False to prevent the code from editing files which are not staged yet
            input_history_file=input_history_file,
            chat_history_file=chat_history_file,
        )

        # INNITIALIIZE AIDER
        aider_coder = Coder.create(
            main_model=self.model,
            fnames=fnames,
            auto_lint=auto_lint,
            #auto_test=auto_test, #manually test for now (since the test_cmd will change for each file)
            lint_cmds={"python": lint_cmd},
            #test_cmd=test_cmd,
            io=io,
        )

        # aider IMPLEMENTS the file 
        aider_coder.run(implement_message)
        
        with open(log_dir / "coding-agent-log-file.txt", "a+") as f:
            f.write("fnames ")
            json.dump(fnames, f)
            f.write("\n")
            json.dump(test_files, f)
            f.write(f"\nlength {len(test_files)}\n")
        
        # Get the base name (e.g., "tensor_data.py")
        base_name = os.path.basename(fnames[0])

        # Split the file name and extension
        file_name, _ = os.path.splitext(base_name)

        # logging for debug
        with open(log_dir / "coding-agent-log-file.txt", "a+") as f:
            f.write(f"implementing {file_name}")

        # DEBUGGING CODE
        for test_file in test_files:                
            if file_name in test_file:
                # logging for debug
                with open(log_dir / "coding-agent-log-file.txt", "a+") as f:
                    f.write(f"\n{file_name} is in {test_file}\n")
                
                n = 1
                for _ in range(n): # try to fix the test case errors in a file n times
                                
                    test_cmd = f"python -m commit0 test {repo_name} {test_file} " + test_cmd_second_half

                    # NOTE trying to run the shell command not through aider
                    # to see if that removes my issue with infinite running software
                    result = subprocess.run(
                        test_cmd,
                        shell=True,
                        text=True,
                        stderr=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        capture_output=True
                    )
                    test_errors = result.stderr

                    # runs test_cmd on shell and adds output to the chat
                    #test_errors = aider_coder.commands.cmd_test(test_cmd)

                    aider_coder.run(f"Modify or rewrite the functions implemented in the file {fnames} " +
                                    f"to resolve the failed unit tests." +
                                    f"The unit test output is: \n {test_errors}\n\n")

                    # NOTE code for askign aider to edit code unit test case by unit test case is commented out below
                    # test output for each test case
                    # header_pattern = r"_{4,} .+ _{4,}"
                    # split_sections = re.split(header_pattern, test_errors)        
                                    
                    ## logging debug
                    # with open(log_dir / "coding-agent-log-file.txt", "a+") as f:
                    #     f.write(f"\n{test_file} split sections: ")
                    #     json.dump(split_sections, f)
                    #     f.write("\n")

                    # for test_out in split_sections[1:]:
                    #     if True:# if "FAILED" not in test_out and "FFF" not in test_out:
                    #         aider_coder.run(f"Modify or redo the functions just implemented in the file {fnames} " +
                    #                 f"to resolve the following failed unit test for your " +
                    #                 f"implementation. The unit test output is: \n {test_out}\n\n")
                    #                 # f"If the failed unit test is not relevant to the functions in the files: {fnames}, then ignore this command and do nothing. Do not add any new files to chat." +
                    #                 # f"The unit test failed is in the file {test_file}."
                                        
        
        sys.stdout.close()
        sys.stderr.close()
        # Restore original stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        return AiderReturn(log_file)