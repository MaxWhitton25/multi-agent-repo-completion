import os
import sys
import yaml
import multiprocessing
import bz2
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

from custom_agent_utils import parse_tasks, get_test_results_json, revert_to_commit
from map import get_test_mapping

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
    _, repo_name = example["repo"].split("/")

    # before starting, display all information to terminal
    update_queue.put(("start_repo", (repo_name, 0)))

    # repo_name = repo_name.lower()
    # repo_name = repo_name.replace(".", "-")

    repo_path = os.path.join(repo_base_dir, repo_name)
    repo_path = os.path.abspath(repo_path)
    
    try:
        local_repo = Repo(repo_path)
    except Exception:
        raise Exception(
            f"{repo_path} is not a git repo. Check if base_dir is correctly specified."
        )
        
    test_mapping = get_test_mapping(Path(repo_path))
    #manager_agent = ManagerAgent(1, agent_config.model_name)
    coder_agent = DebugAgent(agent_config.max_iteration, agent_config.model_name, test_mapping, local_repo)
    coder_agent.setup(repo_name, branch, commit0_config_file)

    # Check if there are changes in the current branch
    if local_repo.is_dirty():
        # Stage all changes
        local_repo.git.add(A=True)
        # Commit changes with the message "left from last change"
        local_repo.index.commit("left from last change")

    # # if branch_name is not provided, create a new branch name based on agent_config
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

    lint_files = get_changed_files_from_commits(
        local_repo, "HEAD", example["base_commit"]
    )
    # Call the commit0 get-tests command to retrieve test files
    test_files_str = [xx for x in get_tests(repo_name, verbose=0) for xx in x]
    test_files = sorted(list(set([i.split(":")[0] for i in test_files_str])))

    # prepare the log dir
    experiment_log_dir = (
        Path(log_dir)
        / repo_name
        / branch
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    experiment_log_dir.mkdir(parents=True, exist_ok=True)

    eval_results = {}
    # write agent_config to .agent.yaml in the log_dir for record
    agent_config_log_file = experiment_log_dir / ".agent.yaml"
    with open(agent_config_log_file, "w") as agent_config_file:
        yaml.dump(agent_config, agent_config_file)
                
    manager_message = f"""You are a manager in charge of writing a plan to complete the implementations for all functions (i.e., those with pass statements) and pass the unit tests. Write a plan of attack to implement the entire repo, keeping in mind the most effective order in which tasks should be implemented. Please output the plan in the format of a list of numbered steps. Each step should specify a file to edit and a high-level description of the change to make. Note that we only need to edit the files that contain functions with pass statements, ie. those in the current context. Give me ONLY the plan, with no extraneous text.
    
    You MUST precede the plan with the keyword PLAN_START, and end it with the keyword PLAN_END. You MUST follow the formatting of the example plan below, with a number preceding each step on a new line, and one file name followed by a colon and a detailed description of the change to make:
    
    PLAN_START
    1.) example_file.py: description of function(s) to implement in example_file.py, including any relevant context or dependencies
    2.) example_file2.py: description of function(s) to implement in example_file2.py, including any relevant context or dependencies
    ... 
    PLAN_END
    
    Remember that you must modify all of the target edit files: {target_edit_files}
    The plan does not neccessarily need to edit the whole file in one step, and it may be more granular as you see fit. Keep in mind that the order in which the files/functions are implemented are very important; make sure that no functions' dependencies are being implemented before the function itself. You should look at the file 'spec.pdf' for more information on the project requirements and specifications.
    """

    with DirContext(repo_path):
        if agent_config is None:
            raise ValueError("Invalid input")

        else:
            
            message = get_message(agent_config, repo_path, test_files=test_files)

            update_queue.put(("start_repo", (repo_name, len(target_edit_files))))
            for f in target_edit_files:
                update_queue.put(("set_current_file", (repo_name, f)))
                if agent_config.add_import_module_to_context:
                    dependencies = import_dependencies.get(f, [])
                    message = update_message_with_dependencies(message, dependencies)
                    
                file_name = f.replace(".py", "").replace("/", "__")
                file_log_dir = experiment_log_dir / file_name
                lint_cmd = get_lint_cmd(
                    repo_name, agent_config.use_lint_info, commit0_config_file
                )
                
                agent_return = coder_agent.run(message, lint_cmd, [f], file_log_dir, repo_name, branch, commit0_config_file)
                update_queue.put(
                    (
                        "update_money_display",
                        (repo_name, file_name, agent_return.last_cost),
                    )
                )
            
            '''
            file_name = "all"
            file_log_dir = experiment_log_dir / file_name
            lint_cmd = get_lint_cmd(repo_name, agent_config.use_lint_info, commit0_config_file)
            
            agent_return = manager_agent.run(manager_message, target_edit_files, file_log_dir)
            
            update_queue.put(
                (
                    "update_money_display",
                    (repo_name, file_name, agent_return.last_cost),
                )
            )
                        
            with open(agent_return.log_file, 'r', encoding='utf-8') as file:
                plan = file.read()
            
            tasks = parse_tasks(plan)
            
            for file_name, description in tasks:
                update_queue.put(("set_current_file", (repo_name, file_name)))
                
                implement_message = f"Complete the following task, implementing the relevant incomplete function(s) (i.e., those with pass statements): \n{description}"
                #TODO: fix the display (right now it just displys one file)
                
                second_half_of_test_cmd = f"--branch {branch} --commit0-config-file {commit0_config_file} --timeout 100"

                
                #TODO: MAKE THE DEBUG/CODER AGENT IMPLEMENT THE ORIGINAL TASK
                agent_return = coder_agent.run(implement_message, 
                                               second_half_of_test_cmd, 
                                               lint_cmd, 
                                               [file_name], 
                                               file_log_dir, 
                                               repo_name=repo_name)
                
                #TODO: MAKE THE DEBUG/CODER AGENT DEBUG THE IMPLEMENTATION

                update_queue.put(
                    (
                        "update_money_display",
                        (repo_name, file_name, agent_return.last_cost),
                    )
                ) '''
                
    if agent_config.record_test_for_each_commit:
        with open(experiment_log_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f)

    update_queue.put(("finish_repo", repo_name))


class DebugAgent(AiderAgents):
    
    def __init__(self, max_iteration: int, model_name: str, test_mapping: dict, local_repo: Repo):
        super().__init__(max_iteration, model_name)
        self.test_mapping = test_mapping
        self.local_repo = local_repo
        self.prev_test_results = None
        
    def setup(self, repo_name: str, branch: str, commit0_config_file: str) -> None:
        self.prev_test_results = get_test_results_json(repo_name, branch, commit0_config_file)
        
    def are_equivalent_errors(self, e1: dict, e2: dict) -> bool:
        """Compares two error messages, ignoring hex addresses like '0x...' regardless of context."""
        
        hex_pattern = r"0x[0-9a-fA-F]+"
        
        s1 = e1["call"]["crash"]["message"]
        s2 = e2["call"]["crash"]["message"]
        
        # Replace all hex addresses with placeholder
        clean_s1 = re.sub(hex_pattern, "<hex>", s1)
        clean_s2 = re.sub(hex_pattern, "<hex>", s2)
        
        return clean_s1 == clean_s2
    
    def find_unique_errors_for_file(self, test_results: dict, file: str) -> list[str]:
        """Find unique errors in the current test results for the given file."""
        
        related_errors = []
        
        # Find the stemmed name of the file (ex. src/file.py -> file)
        file_stem = Path(file).stem
        
        for curr_test in test_results["tests"]:
            if curr_test["outcome"] != "passed":
                # Remove things in brackets (ex. tests/test_resources.py::test_initial_store_capacity[Store]) to get the test case name
                test_case = re.sub(r"\[.*?\]", "", curr_test["nodeid"])
                file_to_functions = self.test_mapping[test_case]
                
                # Check if the file name is in the test case name
                if file_stem in test_case:
                    related_errors.append(curr_test)
                    continue
                
                # Check if the file name is in the mapping
                for test_file, functions in file_to_functions.items():
                    if test_file in file:
                        related_errors.append(curr_test)
                        break
                            
        unique_errors = [] 
        for curr_test in related_errors:
            if all(
                not self.are_equivalent_errors(curr_test, unique_error) for unique_error in unique_errors
                ):
                unique_errors.append(curr_test)
                
        unique_errors = [f"The unit test failed is {curr_test['nodeid']}\n\n" + \
                        f"The unit test output is: \n {curr_test['call']['longrepr']}\n\n" + \
                        f"The traceback of the unit test failed is: \n {curr_test['call']['traceback']}" for curr_test in unique_errors]
        return unique_errors
    
    def run(
        self,
        implement_message: str,
        lint_cmd: str,
        fnames: list[str],
        log_dir: Path,
        repo_name: str,
        branch: str,
        commit0_config_file: str,
    ) -> AgentReturn:
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
        
        # find hash of the latest commit
        baseline_commit = self.local_repo.head.commit.hexsha

        io = InputOutput(
            yes=False,
            input_history_file=input_history_file,
            chat_history_file=chat_history_file,
        )

        # INNITIALIIZE AIDER
        coder = Coder.create(
            main_model=self.model,
            fnames=fnames,
            auto_lint=auto_lint,
            #auto_test=auto_test, #manually test for now (since the test_cmd will change for each file)
            lint_cmds={"python": lint_cmd},
            #test_cmd=test_cmd,
            io=io,
        )

        # IMPLEMENTATION CODE
        coder.run(implement_message)
        
        attempts = 0
        while attempts < self.max_iteration:
        
            test_results = None
            error_info = None
            try:
                test_results = get_test_results_json(repo_name, branch, commit0_config_file)
            except RuntimeError as e:
                error_info = e
            
            if test_results:
                # check if the number of passed tests has decreased
                if test_results["summary"]["passed"] <= self.prev_test_results["summary"]["passed"]:
                    error_info = "\n".join(self.find_unique_errors_for_file(test_results, fnames[0]))     
                    
            if error_info:
                coder.run("/reset")
                coder.run(f"/add {fnames[0]}")
                
                prompt = implement_message + f"\n\nYour previous attempt to implement this file resulted in a decrease in the number of passed tests. When implementing this file, please keep in mind the following error output .\n\n{error_info}"
                
                revert_to_commit(self.local_repo, baseline_commit)
                
                coder.run(prompt)
                attempts += 1
            else:
                break
                
        self.prev_test_results = test_results
                            
        sys.stdout.close()
        sys.stderr.close()
        # Restore original stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        return AiderReturn(log_file)


class ManagerAgent(AiderAgents):
        
    def run(
        self,
        message: str,
        fnames: list[str],
        log_dir: Path,
    ) -> AgentReturn:
        """Start agent manager"""
        
        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        input_history_file = log_dir / ".manager.input.history"
        chat_history_file = log_dir / ".manager.chat.history.md"

        # Set up logging
        log_file = log_dir / "manager.log"
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
        
        # Get the specifications
        with bz2.open("spec.pdf.bz2", "rb") as in_file:
            with open("spec.pdf", "wb") as out_file:
                out_file.write(in_file.read())

        io = InputOutput(
            yes=False,
            input_history_file=input_history_file,
            chat_history_file=chat_history_file,
        )
        manager = Coder.create(
            edit_format="ask",
            main_model=self.model,
            read_only_fnames=fnames + ["spec.pdf"],
            io=io,
        )
        manager.max_reflection = self.max_iteration
        manager.stream = True
        
        manager.run(message)

        sys.stdout.close()
        sys.stderr.close()
        # Restore original stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        return AiderReturn(log_file)

# class RollbackAgents(AiderAgents):
#     def run(
#         self,
#         message: str,
#         test_cmd: str,
#         lint_cmd: str,
#         fnames: list[str],
#         log_dir: Path,
#         test_first: bool = False,
#         lint_first: bool = False,
#     ) -> AgentReturn:
#         """Start aider agent"""
#         if test_cmd:
#             auto_test = True
#         else:
#             auto_test = False
#         if lint_cmd:
#             auto_lint = True
#         else:
#             auto_lint = False
#         log_dir = log_dir.resolve()
#         log_dir.mkdir(parents=True, exist_ok=True)
#         input_history_file = log_dir / ".aider.input.history"
#         chat_history_file = log_dir / ".aider.chat.history.md"

#         # Set up logging
#         log_file = log_dir / "aider.log"
#         logging.basicConfig(
#             filename=log_file,
#             level=logging.INFO,
#             format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#         )

#         # Redirect print statements to the log file
#         sys.stdout = open(log_file, "a")
#         sys.stderr = open(log_file, "a")

#         # Configure httpx and backoff logging
#         handle_logging("httpx", log_file)
#         handle_logging("backoff", log_file)

#         io = InputOutput(
#             yes=True,
#             input_history_file=input_history_file,
#             chat_history_file=chat_history_file,
#         )
#         coder = RollbackCoder.create(
#             main_model=self.model,
#             fnames=fnames,
#             auto_lint=auto_lint,
#             auto_test=auto_test,
#             lint_cmds={"python": lint_cmd},
#             test_cmd=test_cmd,
#             io=io,
#         )
#         coder.max_reflections = self.max_iteration
#         coder.stream = True

#         # Run the agent
#         if test_first:
#             test_errors = coder.commands.cmd_test(test_cmd)
#             if test_errors:
#                 coder.run(test_errors)
#         elif lint_first:
#             coder.commands.cmd_lint(fnames=fnames)
#         else:
#             coder.run(message)

#         # Close redirected stdout and stderr
#         sys.stdout.close()
#         sys.stderr.close()
#         # Restore original stdout and stderr
#         sys.stdout = sys.__stdout__
#         sys.stderr = sys.__stderr__

#         return AgentReturn(log_file)
    
