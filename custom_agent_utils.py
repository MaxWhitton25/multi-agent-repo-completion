import subprocess
import re
from typing import List
import git

def parse_tasks(text: str) -> list[tuple[str, str]]:
    """Parse the tasks from the manager output."""
    tasks = []

    # Extract the portion between PLAN_START and PLAN_END
    plan_match = re.search(r"PLAN_START(.*?)PLAN_END", text, re.DOTALL)
    if not plan_match:
        return tasks  

    # Get the plan content and split by patterns that indicate the start of a new task
    plan_text = plan_match.group(1).strip()
    task_blocks = re.split(r"\d+\.?\)?\s+", plan_text)  # Split at task numbers

    for block in task_blocks:
        if not block.strip():
            continue
        # Match the file name and task description
        match = re.search(r"([\w\-/\.0-9]+\.\w+):\s*(.*)", block.strip(), re.DOTALL)
        if match:
            file_name = match.group(1)
            description = re.sub(r'\s+', ' ', match.group(2)).strip()  # Remove extra spaces
            tasks.append((file_name, description))

    return tasks


"""
Helper functions for Revert code
"""
# def split_into_batches(files: List[str], batch_size: int) -> List[List[str]]:
#     """Split the list of files into batches of given size."""
#     return [files[i:i + batch_size] for i in range(0, len(files), batch_size)]

def check_performance_improved(previous_results: dict, current_results: dict) -> bool:
    """Compare previous and current test results to determine if performance has improved."""
    previous_passed = previous_results.get('num_passed', 0)
    current_passed = current_results.get('num_passed', 0)
    return current_passed > previous_passed


def revert_to_commit(repo: git.Repo, commit_hash: str) -> None:
    """Revert the repository to the specified commit hash."""
    repo.git.reset('--hard', commit_hash)

# def get_message_for_batch(
#     agent_config: AgentConfig,
#     repo_path: str,
#     function_batch: List[str],
#     test_files: List[str],
# ) -> str:
#     """Generate the message for the agent for the current batch of functions."""
#     message = get_message(agent_config, repo_path, test_files)
#     functions_info = "\n".join(function_batch)
#     message += f"\n\n>>> Focus on implementing the following functions:\n{functions_info}"
#     return message