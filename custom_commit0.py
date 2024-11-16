import sys
import os
import bz2

from pathlib import Path
import logging

from aider.io import InputOutput
from agent.class_types import AgentConfig
from agent.run_agent import DirContext
from agent.agents import AiderAgents, AgentReturn, AiderReturn, handle_logging
from aider.coders import Coder


from commit0.harness.constants import SPLIT
from commit0.harness.get_pytest_ids import main as get_tests
from commit0.harness.constants import RUN_AGENT_LOG_DIR, RepoInstance
from pathlib import Path
from datetime import datetime
#from custom_base_coder import RollbackCoder

import yaml
import multiprocessing
from datasets import load_dataset
from git import Repo
from agent.agent_utils import (
    create_branch,
    get_message,
    get_target_edit_files,
    update_message_with_dependencies,
    get_lint_cmd,
    read_yaml_config,
)

class RollbackAgents(AiderAgents):
    def run(
        self,
        message: str,
        test_cmd: str,
        lint_cmd: str,
        fnames: list[str],
        log_dir: Path,
        test_first: bool = False,
        lint_first: bool = False,
    ) -> AgentReturn:
        """Start aider agent"""
        if test_cmd:
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

        io = InputOutput(
            yes=True,
            input_history_file=input_history_file,
            chat_history_file=chat_history_file,
        )
        coder = RollbackCoder.create(
            main_model=self.model,
            fnames=fnames,
            auto_lint=auto_lint,
            auto_test=auto_test,
            lint_cmds={"python": lint_cmd},
            test_cmd=test_cmd,
            io=io,
        )
        coder.max_reflections = self.max_iteration
        coder.stream = True

        # Run the agent
        if test_first:
            test_errors = coder.commands.cmd_test(test_cmd)
            if test_errors:
                coder.run(test_errors)
        elif lint_first:
            coder.commands.cmd_lint(fnames=fnames)
        else:
            coder.run(message)

        # Close redirected stdout and stderr
        sys.stdout.close()
        sys.stderr.close()
        # Restore original stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        return AgentReturn(log_file)
    
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
    
def custom_run_agent_for_repo(
    repo_base_dir: str,
    agent_config: AgentConfig,
    example: RepoInstance,
    update_queue: multiprocessing.Queue,
    branch: str,
    override_previous_changes: bool = False,
    backend: str = "modal",
    log_dir: str = str(RUN_AGENT_LOG_DIR.resolve()),
) -> None:
    """Run Aider for a given repository."""
    # get repo info
    _, repo_name = example["repo"].split("/")

    # before starting, display all information to terminal
    original_repo_name = repo_name
    update_queue.put(("start_repo", (original_repo_name, 0)))

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

    if agent_config.agent_name == "aider":
        agent = AiderAgents(agent_config.max_iteration, agent_config.model_name)
    elif agent_config.agent_name == "aider-rollback":
        agent = RollbackAgents(agent_config.max_iteration, agent_config.model_name)
    else:
        raise NotImplementedError(
            f"{agent_config.agent_name} is not implemented; please add your implementations in baselines/agents.py."
        )

    # # if branch_name is not provided, create a new branch name based on agent_config
    # if branch is None:
    #     branch = args2string(agent_config)
    create_branch(local_repo, branch, example["base_commit"])

    # in cases where the latest commit of branch is not commit 0
    # set it back to commit 0
    latest_commit = local_repo.commit(branch)
    if latest_commit.hexsha != example["base_commit"] and override_previous_changes:
        local_repo.git.reset("--hard", example["base_commit"])

    target_edit_files, import_dependencies = get_target_edit_files(
        local_repo,
        example["src_dir"],
        example["test"]["test_dir"],
        str(latest_commit),
        example["reference_commit"],
    )
    # Call the commit0 get-tests command to retrieve test files
    test_files_str = get_tests(repo_name, verbose=0)
    test_files = sorted(list(set([i.split(":")[0] for i in test_files_str])))

    # prepare the log dir
    experiment_log_dir = (
        Path(log_dir)
        / repo_name
        / branch
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    experiment_log_dir.mkdir(parents=True, exist_ok=True)

    # write agent_config to .agent.yaml in the log_dir for record
    agent_config_log_file = experiment_log_dir / ".agent.yaml"
    with open(agent_config_log_file, "w") as agent_config_file:
        yaml.dump(agent_config, agent_config_file)

    # TODO: make this path more general
    commit0_dot_file_path = str(Path(repo_path).parent.parent / ".commit0.yaml")

    with DirContext(repo_path):
        if agent_config is None:
            raise ValueError("Invalid input")

        if agent_config.run_tests:
            update_queue.put(("start_repo", (original_repo_name, len(test_files))))
            # when unit test feedback is available, iterate over test files
            for test_file in test_files:
                update_queue.put(("set_current_file", (repo_name, test_file)))
                test_cmd = f"python -m commit0 test {repo_path} {test_file} --branch {branch} --backend {backend} --commit0-dot-file-path {commit0_dot_file_path}"
                test_file_name = test_file.replace(".py", "").replace("/", "__")
                test_log_dir = experiment_log_dir / test_file_name
                lint_cmd = get_lint_cmd(repo_name, agent_config.use_lint_info)
                message = get_message(agent_config, repo_path, test_file=test_file)

                # display the test file to terminal
                agent_return = agent.run(
                    message,
                    test_cmd,
                    lint_cmd,
                    target_edit_files,
                    test_log_dir,
                    test_first=True,
                )
                # after running the agent, update the money display
                update_queue.put(
                    (
                        "update_money_display",
                        (repo_name, test_file, agent_return.last_cost),
                    )
                )
        else:
            # when unit test feedback is not available, iterate over target files to edit
            message = get_message(
                agent_config, repo_path, test_dir=example["test"]["test_dir"]
            )

            update_queue.put(
                ("start_repo", (original_repo_name, len(target_edit_files)))
            )
            for f in target_edit_files:
                update_queue.put(("set_current_file", (repo_name, f)))
                dependencies = import_dependencies[f]
                message = update_message_with_dependencies(message, dependencies)
                file_name = f.replace(".py", "").replace("/", "__")
                file_log_dir = experiment_log_dir / file_name
                lint_cmd = get_lint_cmd(repo_name, agent_config.use_lint_info)
                agent_return = agent.run(message, "", lint_cmd, [f], file_log_dir)
                update_queue.put(
                    (
                        "update_money_display",
                        (repo_name, file_name, agent_return.last_cost),
                    )
                )
    update_queue.put(("finish_repo", original_repo_name))

### VERSION OF CUSTOM_RUN_AGENT_FOR_REPO which is up to date with git, not pip (11/12)
# def custom_run_agent_for_repo(
#     repo_base_dir: str,
#     agent_config: AgentConfig,
#     example: RepoInstance,
#     branch: str,
#     update_queue: multiprocessing.Queue,
#     override_previous_changes: bool = False,
#     backend: str = "modal",
#     log_dir: str = str(RUN_AGENT_LOG_DIR.resolve()),
#     commit0_config_file: str = "",
# ) -> None:
#     """Run Aider for a given repository."""
#     # get repo info
#     commit0_config = read_commit0_config_file(commit0_config_file)

#     assert "commit0" in commit0_config["dataset_name"]
#     _, repo_name = example["repo"].split("/")

#     # before starting, display all information to terminal
#     update_queue.put(("start_repo", (repo_name, 0)))

#     # repo_name = repo_name.lower()
#     # repo_name = repo_name.replace(".", "-")

#     repo_path = os.path.join(repo_base_dir, repo_name)
#     repo_path = os.path.abspath(repo_path)

#     try:
#         local_repo = Repo(repo_path)
#     except Exception:
#         raise Exception(
#             f"{repo_path} is not a git repo. Check if base_dir is correctly specified."
#         )

#     if agent_config.agent_name == "aider":
#         agent = AiderAgents(agent_config.max_iteration, agent_config.model_name)
#     elif agent_config.agent_name == "aider-rollback":
#         agent = RollbackAgents(agent_config.max_iteration, agent_config.model_name)
#     else:
#         raise NotImplementedError(
#             f"{agent_config.agent_name} is not implemented; please add your implementations in baselines/agents.py."
#         )

#     # Check if there are changes in the current branch
#     if local_repo.is_dirty():
#         # Stage all changes
#         local_repo.git.add(A=True)
#         # Commit changes with the message "left from last change"
#         local_repo.index.commit("left from last change")

#     # # if branch_name is not provided, create a new branch name based on agent_config
#     # if branch is None:
#     #     branch = args2string(agent_config)
#     create_branch(local_repo, branch, example["base_commit"])

#     # in cases where the latest commit of branch is not commit 0
#     # set it back to commit 0
#     latest_commit = local_repo.commit(branch)
#     if latest_commit.hexsha != example["base_commit"] and override_previous_changes:
#         local_repo.git.reset("--hard", example["base_commit"])

#     # get target files to edit and test files to run
#     target_edit_files, import_dependencies = get_target_edit_files(
#         local_repo,
#         example["src_dir"],
#         example["test"]["test_dir"],
#         branch,
#         example["reference_commit"],
#         agent_config.use_topo_sort_dependencies,
#     )

#     lint_files = get_changed_files_from_commits(
#         local_repo, "HEAD", example["base_commit"]
#     )
#     # Call the commit0 get-tests command to retrieve test files
#     test_files_str = get_tests(repo_name, verbose=0)
#     test_files = sorted(list(set([i.split(":")[0] for i in test_files_str])))

#     # prepare the log dir
#     experiment_log_dir = (
#         Path(log_dir)
#         / repo_name
#         / branch
#         / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     )
#     experiment_log_dir.mkdir(parents=True, exist_ok=True)

#     eval_results = {}
#     # write agent_config to .agent.yaml in the log_dir for record
#     agent_config_log_file = experiment_log_dir / ".agent.yaml"
#     with open(agent_config_log_file, "w") as agent_config_file:
#         yaml.dump(agent_config, agent_config_file)

#     with DirContext(repo_path):
#         if agent_config is None:
#             raise ValueError("Invalid input")

#         if agent_config.run_tests:
#             update_queue.put(("start_repo", (repo_name, len(test_files))))
#             # when unit test feedback is available, iterate over test files
#             for test_file in test_files:
#                 update_queue.put(("set_current_file", (repo_name, test_file)))
#                 test_cmd = f"python -m commit0 test {repo_path} {test_file} --branch {branch} --backend {backend} --commit0-config-file {commit0_config_file} --timeout 100"
#                 test_file_name = test_file.replace(".py", "").replace("/", "__")
#                 test_log_dir = experiment_log_dir / test_file_name
#                 lint_cmd = get_lint_cmd(
#                     repo_name, agent_config.use_lint_info, commit0_config_file
#                 )
#                 message = get_message(agent_config, repo_path, test_files=[test_file])

#                 # display the test file to terminal
#                 agent_return = agent.run(
#                     "",
#                     test_cmd,
#                     lint_cmd,
#                     target_edit_files,
#                     test_log_dir,
#                     test_first=True,
#                 )
#                 if agent_config.record_test_for_each_commit:
#                     current_commit = local_repo.head.commit.hexsha
#                     eval_results[current_commit] = run_eval_after_each_commit(
#                         branch, backend, commit0_config_file
#                     )

#                 # after running the agent, update the money display
#                 update_queue.put(
#                     (
#                         "update_money_display",
#                         (repo_name, test_file, agent_return.last_cost),
#                     )
#                 )
#         elif agent_config.run_entire_dir_lint:
#             update_queue.put(("start_repo", (repo_name, len(lint_files))))
#             # when unit test feedback is available, iterate over test files
#             for lint_file in lint_files:
#                 update_queue.put(("set_current_file", (repo_name, lint_file)))
#                 lint_file_name = lint_file.replace(".py", "").replace("/", "__")
#                 lint_log_dir = experiment_log_dir / lint_file_name
#                 lint_cmd = get_lint_cmd(
#                     repo_name, agent_config.use_lint_info, commit0_config_file
#                 )

#                 # display the test file to terminal
#                 agent_return = agent.run(
#                     "",
#                     "",
#                     lint_cmd,
#                     [lint_file],
#                     lint_log_dir,
#                     lint_first=True,
#                 )
#                 if agent_config.record_test_for_each_commit:
#                     current_commit = local_repo.head.commit.hexsha
#                     eval_results[current_commit] = run_eval_after_each_commit(
#                         branch, backend, commit0_config_file
#                     )

#                 # after running the agent, update the money display
#                 update_queue.put(
#                     (
#                         "update_money_display",
#                         (repo_name, lint_file, agent_return.last_cost),
#                     )
#                 )
#         else:
#             # when unit test feedback is not available, iterate over target files to edit
#             message = get_message(agent_config, repo_path, test_files=test_files)

#             update_queue.put(("start_repo", (repo_name, len(target_edit_files))))
#             for f in target_edit_files:
#                 update_queue.put(("set_current_file", (repo_name, f)))
#                 if agent_config.add_import_module_to_context:
#                     dependencies = import_dependencies.get(f, [])
#                     message = update_message_with_dependencies(message, dependencies)
#                 file_name = f.replace(".py", "").replace("/", "__")
#                 file_log_dir = experiment_log_dir / file_name
#                 lint_cmd = get_lint_cmd(
#                     repo_name, agent_config.use_lint_info, commit0_config_file
#                 )
#                 agent_return = agent.run(message, "", lint_cmd, [f], file_log_dir)
#                 if agent_config.record_test_for_each_commit:
#                     current_commit = local_repo.head.commit.hexsha
#                     eval_results[current_commit] = run_eval_after_each_commit(
#                         branch, backend, commit0_config_file
#                     )

#                 update_queue.put(
#                     (
#                         "update_money_display",
#                         (repo_name, file_name, agent_return.last_cost),
#                     )
#                 )
#     if agent_config.record_test_for_each_commit:
#         with open(experiment_log_dir / "eval_results.json", "w") as f:
#             json.dump(eval_results, f)

#     update_queue.put(("finish_repo", repo_name))
