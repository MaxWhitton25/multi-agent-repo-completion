import re 
import ast

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


def get_classes_from_file(file_path):
    try:
        with open(file_path, "r") as file:
            tree = ast.parse(file.read())
        
        # Get all class definitions from the AST
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        return class_names if class_names else None
    except Exception as e:
        return f"Error: {e}"