import re 
import json
import subprocess
from commit0.harness.utils import get_hash_string
from commit0.harness.constants import RUN_PYTEST_LOG_DIR

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

def get_test_results_json(repo_name: str, branch: str, commit0_config_file: str) -> dict:
    # Assuming the test directory is named "tests"
    test_ids = "tests"
    
    command = [
        "python", "-m", "commit0", "test", repo_name, test_ids, "--branch", branch, "--commit0-config-file", commit0_config_file, "--timeout", "100"
    ]
    
    hashed_test_ids = get_hash_string(test_ids)
    # set up logging
    log_dir = RUN_PYTEST_LOG_DIR / repo_name / branch / hashed_test_ids
    report_file = log_dir / "report.json"
    
    try: 
        subprocess.run(command, check=True)
    # Called when any pytest tests fail
    except subprocess.CalledProcessError as e:
        pass
    
    try:
        with open(report_file, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        raise Exception(f"Test report file not found: {report_file}")