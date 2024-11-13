import os
import yaml
import multiprocessing
from tqdm import tqdm
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
    split_into_batches, # Revert
    check_performance_improved, # Revert
    revert_to_commit, # Revert
)
import subprocess
import json
from agent.agents import AiderAgents
from typing import cast
from agent.class_types import AgentConfig
from commit0.harness.constants import SPLIT
from commit0.harness.get_pytest_ids import main as get_tests
from commit0.harness.constants import RUN_AGENT_LOG_DIR, RepoInstance
from commit0.cli import read_commit0_config_file
from pathlib import Path
from datetime import datetime
from agent.run_agent import DirContext, run_eval_after_each_commit
from typing import List # Revert


# Revert
def get_message_for_batch(
    agent_config: AgentConfig,
    repo_path: str,
    function_batch: List[str],
    test_files: List[str],
) -> str:
    """Generate the message for the agent for the current batch of functions."""
    message = get_message(agent_config, repo_path, test_files)
    functions_info = "\n".join(function_batch)
    message += f"\n\n>>> Focus on implementing the following functions:\n{functions_info}"
    return message


def run_agent_for_repo(
    repo_base_dir: str,
    agent_config: AgentConfig,
    example: RepoInstance,
    branch: str,
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

    # repo_name = repo_name.lower()
    # repo_name = repo_name.replace(".", "-")
    repo_name = example["repo"].split("/")[-1] # Revert

    repo_path = os.path.join(repo_base_dir, repo_name)
    repo_path = os.path.abspath(repo_path)

    try:
        local_repo = Repo(repo_path)
    except Exception:
        raise Exception(
            f"{repo_path} is not a git repo. Check if base_dir is correctly specified."
        )

    if agent_config.agent_name == "aider":
        agent = AiderAgents(agent_config.max_iteration, agent_config.model_name, agent_config, local_repo) # Revert
    else:
        raise NotImplementedError(
            f"{agent_config.agent_name} is not implemented; please add your implementations in baselines/agents.py."
        )

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

    # Initialize baseline commit
    baseline_commit = local_repo.head.commit.hexsha

    # Run initial evaluation to get baseline performance
    initial_eval_results = run_eval_after_each_commit(
        branch, backend, commit0_config_file
    )
    initial_eval_results_list = json.loads(initial_eval_results)
    for result in initial_eval_results_list:
        if result['name'] == repo_name:
            initial_num_passed_tests = result.get('num_passed', 0)
            total_tests = result.get('num_tests', 0)
            break
    else:
        initial_num_passed_tests = 0
        total_tests = 0
    previous_results = {'num_passed': initial_num_passed_tests, 'num_tests': total_tests}

    # Split target files into batches
    function_batches = split_into_batches(
        target_edit_files,
        agent_config.incremental_step_size,
    )

    # Iterate over batches
    for function_batch in function_batches:
        retries = 0
        success = False
        while retries < agent_config.max_retries and not success:
            # Update message and dependencies for the current batch
            current_message = get_message_for_batch(
                agent_config, repo_path, function_batch, test_files
            )

            # Run the agent
            agent_return = agent.run(
                current_message,
                "",  # test_cmd
                "",  # lint_cmd
                function_batch,
                experiment_log_dir,
                test_first=False,
                lint_first=False,
            )

            # Run tests and check performance
            current_eval_results = run_eval_after_each_commit(
                branch, backend, commit0_config_file
            )
            current_eval_results_list = json.loads(current_eval_results)
            for result in current_eval_results_list:
                if result['name'] == repo_name:
                    current_num_passed_tests = result.get('num_passed', 0)
                    total_tests = result.get('num_tests', 0)
                    break
            else:
                current_num_passed_tests = 0
                total_tests = 0
            current_results = {'num_passed': current_num_passed_tests, 'num_tests': total_tests}

            performance_improved = check_performance_improved(
                previous_results, current_results
            )

            if performance_improved:
                # Keep changes
                baseline_commit = local_repo.head.commit.hexsha
                previous_results = current_results
                success = True
            else:
                # Revert changes
                revert_to_commit(local_repo, baseline_commit)
                retries += 1

        if not success:
            # Skip to next batch
            continue

    eval_results = {}

    # write agent_config to .agent.yaml in the log_dir for record
    agent_config_log_file = experiment_log_dir / ".agent.yaml"
    with open(agent_config_log_file, "w") as agent_config_file:
        yaml.dump(agent_config, agent_config_file)

    with DirContext(repo_path):
        if agent_config is None:
            raise ValueError("Invalid input")

        if agent_config.run_tests:
            # when unit test feedback is available, iterate over test files
            for test_file in test_files:
                test_cmd = f"python -m commit0 test {repo_path} {test_file} --branch {branch} --backend {backend} --commit0-config-file {commit0_config_file} --timeout 100"
                test_file_name = test_file.replace(".py", "").replace("/", "__")
                test_log_dir = experiment_log_dir / test_file_name
                lint_cmd = get_lint_cmd(
                    repo_name, agent_config.use_lint_info, commit0_config_file
                )
                message = get_message(agent_config, repo_path, test_files=[test_file])

                # display the test file to terminal
                _ = agent.run(
                    "",
                    test_cmd,
                    lint_cmd,
                    target_edit_files,
                    test_log_dir,
                    test_first=True,
                )
                if agent_config.record_test_for_each_commit:
                    current_commit = local_repo.head.commit.hexsha
                    eval_results[current_commit] = run_eval_after_each_commit(
                        branch, backend, commit0_config_file
                    )
        elif agent_config.run_entire_dir_lint:
            # when unit test feedback is available, iterate over test files
            for lint_file in lint_files:
                lint_file_name = lint_file.replace(".py", "").replace("/", "__")
                lint_log_dir = experiment_log_dir / lint_file_name
                lint_cmd = get_lint_cmd(
                    repo_name, agent_config.use_lint_info, commit0_config_file
                )

                # display the test file to terminal
                _ = agent.run(
                    "",
                    "",
                    lint_cmd,
                    [lint_file],
                    lint_log_dir,
                    lint_first=True,
                )
                if agent_config.record_test_for_each_commit:
                    current_commit = local_repo.head.commit.hexsha
                    eval_results[current_commit] = run_eval_after_each_commit(
                        branch, backend, commit0_config_file
                    )
        else:
            # when unit test feedback is not available, iterate over target files to edit
            message = get_message(agent_config, repo_path, test_files=test_files)

            for f in target_edit_files:
                if agent_config.add_import_module_to_context:
                    dependencies = import_dependencies.get(f, [])
                    message = update_message_with_dependencies(message, dependencies)
                file_name = f.replace(".py", "").replace("/", "__")
                file_log_dir = experiment_log_dir / file_name
                lint_cmd = get_lint_cmd(
                    repo_name, agent_config.use_lint_info, commit0_config_file
                )
                _ = agent.run(message, "", lint_cmd, [f], file_log_dir)
                if agent_config.record_test_for_each_commit:
                    current_commit = local_repo.head.commit.hexsha
                    eval_results[current_commit] = run_eval_after_each_commit(
                        branch, backend, commit0_config_file
                    )
    if agent_config.record_test_for_each_commit:
        with open(experiment_log_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f)


def run_agent(
    branch: str,
    override_previous_changes: bool,
    backend: str,
    agent_config_file: str,
    commit0_config_file: str,
    log_dir: str,
    max_parallel_repos: int,
) -> None:
    """Main function to run Aider for a given repository.

    Will run in parallel for each repo.
    """
    config = read_yaml_config(agent_config_file)

    agent_config = AgentConfig(**config)

    commit0_config_file = os.path.abspath(commit0_config_file)
    commit0_config = read_commit0_config_file(commit0_config_file)

    dataset = load_dataset(
        commit0_config["dataset_name"], split=commit0_config["dataset_split"]
    )
    filtered_dataset = [
        example
        for example in dataset
        if commit0_config["repo_split"] == "all"
        or (
            isinstance(example, dict)
            and "repo" in example
            and isinstance(example["repo"], str)
            and example["repo"].split("/")[-1]
            in SPLIT.get(commit0_config["repo_split"], [])
        )
    ]
    assert len(filtered_dataset) > 0, "No examples available"

    # if len(filtered_dataset) > 1:
    #     sys.stdout = open(os.devnull, "w")
    if agent_config.add_import_module_to_context:
        # Install Chrome for Playwright for browser-based agents
        try:
            subprocess.run(["playwright", "install", "chromium"], check=True)
            print("Chrome installed successfully for Playwright")
        except subprocess.CalledProcessError as e:
            print(f"Error installing Chrome for Playwright: {e}")
        except FileNotFoundError:
            print("Playwright not found. Make sure it's installed and in your PATH.")

    with tqdm(
        total=len(filtered_dataset), smoothing=0, desc="Running Aider for repos"
    ) as pbar:
        with multiprocessing.Pool(processes=max_parallel_repos) as pool:
            results = []

            # Use apply_async to submit jobs and add progress bar updates
            for example in filtered_dataset:
                result = pool.apply_async(
                    run_agent_for_repo,
                    args=(
                        commit0_config["base_dir"],
                        agent_config,
                        cast(RepoInstance, example),
                        branch,
                        override_previous_changes,
                        backend,
                        log_dir,
                        commit0_config_file,
                    ),
                    callback=lambda _: pbar.update(
                        1
                    ),  # Update progress bar on task completion
                )
                results.append(result)

            for result in results:
                result.wait()
