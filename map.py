"""
map.py

This script maps test cases to the functions they invoke within your libraries.
It performs static analysis to identify functions that need implementation and
maps them to the test cases that call them.

Usage:
    python map.py
"""

import os
import sys
import ast
import fnmatch
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from rich.panel import Panel
except ImportError:
    print("The 'rich' library is required for this script to run.")
    print("Install it using 'pip install rich' and try again.")
    sys.exit(1)

console = Console()

# --------------------- Utility Functions ---------------------


def is_library_dir(directory: Path) -> bool:
    """
    Determines if a given directory is a library by checking for a 'tests' subdirectory.

    Args:
        directory (Path): The directory to check.

    Returns:
        bool: True if 'tests' subdirectory exists, False otherwise.
    """
    return (directory / "tests").is_dir()


def find_libraries(base_dir: Path, recursive: bool = True, exclude_dirs: Set[str] = None) -> List[Path]:
    """
    Finds all library directories within the base directory, excluding specified directories.

    Args:
        base_dir (Path): The base directory to search within.
        recursive (bool): Whether to search recursively for libraries.
        exclude_dirs (Set[str], optional): Set of directory names to exclude. Defaults to None.

    Returns:
        List[Path]: A list of Paths to library directories.
    """
    if exclude_dirs is None:
        exclude_dirs = {".venv", "venv", "env", ".env", "site-packages"}

    libraries = []
    if recursive:
        for dir_path in base_dir.rglob("*"):
            if dir_path.is_dir() and is_library_dir(dir_path):
                if any(part in exclude_dirs for part in dir_path.parts):
                    continue
                libraries.append(dir_path)
    else:
        for item in base_dir.iterdir():
            if item.is_dir() and is_library_dir(item):
                if any(part in exclude_dirs for part in item.parts):
                    continue
                libraries.append(item)
    return libraries


def find_source_files(library_dir: Path, exclude_dirs: Set[str] = None, exclude_files: Set[str] = None) -> List[Path]:
    """
    Recursively collects Python source files from the library directory, excluding specified directories and files.

    Args:
        library_dir (Path): The library directory to scan.
        exclude_dirs (Set[str], optional): Set of directory names to exclude. Defaults to None.
        exclude_files (Set[str], optional): Set of exact filenames to exclude. Defaults to None.

    Returns:
        List[Path]: A list of Paths to Python source files.
    """
    if exclude_dirs is None:
        exclude_dirs = {"tests", "build", "dist", "__pycache__", "venv", ".venv", "env", ".env"}
    if exclude_files is None:
        exclude_files = set()

    source_files = []
    for file_path in library_dir.rglob("*.py"):
        if file_path.name in exclude_files:
            continue
        if any(part in exclude_dirs for part in file_path.parts):
            continue
        source_files.append(file_path)
    return source_files


def find_test_files(library_dir: Path, exclude_dirs: Set[str] = None, exclude_files: Set[str] = None) -> List[Path]:
    """
    Recursively collects Python test files from the library directory, excluding specified directories and files.

    Args:
        library_dir (Path): The library directory to scan.
        exclude_dirs (Set[str], optional): Set of directory names to exclude. Defaults to None.
        exclude_files (Set[str], optional): Set of exact filenames to exclude. Defaults to None.

    Returns:
        List[Path]: A list of Paths to Python test files.
    """
    if exclude_dirs is None:
        exclude_dirs = {"build", "dist", "__pycache__", "venv", ".venv", "env", ".env"}
    if exclude_files is None:
        exclude_files = set()

    test_files = []
    for file_path in library_dir.rglob("test_*.py"):
        if file_path.name in exclude_files:
            continue
        if any(part in exclude_dirs for part in file_path.parts):
            continue
        test_files.append(file_path)
    return test_files


def extract_functions_to_implement(source_files: List[Path]) -> Set[str]:
    """
    Extracts function and method names that raise NotImplementedError or contain only a 'pass' statement from source files.

    Args:
        source_files (List[Path]): List of Python source file paths.

    Returns:
        Set[str]: Set of function/method names that need implementation.
    """
    functions_to_implement = set()

    for file_path in source_files:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                file_content = f.read()
        except (UnicodeDecodeError, FileNotFoundError):
            continue

        try:
            tree = ast.parse(file_content, filename=str(file_path))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check for 'raise NotImplementedError' or 'pass' in the function body
                needs_implementation = False
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.Raise):
                        if isinstance(child.exc, ast.Call):
                            if isinstance(child.exc.func, ast.Name) and child.exc.func.id == "NotImplementedError":
                                needs_implementation = True
                                break
                            elif isinstance(child.exc.func, ast.Attribute) and child.exc.func.attr == "NotImplementedError":
                                needs_implementation = True
                                break
                    elif isinstance(child, ast.Pass):
                        needs_implementation = True
                        break
                if needs_implementation:
                    functions_to_implement.add(node.name)
    return functions_to_implement


def extract_test_cases(test_files: List[Path]) -> Dict[str, Path]:
    """
    Extracts test case identifiers from test files.

    Args:
        test_files (List[Path]): List of Python test file paths.

    Returns:
        Dict[str, Path]: Dictionary mapping test case identifiers to their file paths.
                          Example key: "test_tensor.py::test_create"
    """
    test_cases = {}

    for test_file in test_files:
        try:
            with test_file.open("r", encoding="utf-8") as f:
                file_content = f.read()
        except (UnicodeDecodeError, FileNotFoundError):
            continue

        try:
            tree = ast.parse(file_content, filename=str(test_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                identifier = f"{test_file.name}::{node.name}"
                test_cases[identifier] = test_file
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                for child in node.body:
                    if isinstance(child, ast.FunctionDef) and child.name.startswith("test_"):
                        identifier = f"{test_file.name}::{node.name}::{child.name}"
                        test_cases[identifier] = test_file
    return test_cases


def map_tests_to_functions(test_cases: Dict[str, Path], functions_to_implement: Set[str]) -> Dict[str, Set[str]]:
    """
    Maps test cases to functions they invoke by parsing the test file AST.

    Args:
        test_cases (Dict[str, Path]): Dictionary mapping test case identifiers to test file paths.
        functions_to_implement (Set[str]): Set of function/method names that need implementation.

    Returns:
        Dict[str, Set[str]]: Dictionary mapping test case identifiers to the set of functions they invoke.
    """
    mapping = defaultdict(set)

    for test_id, test_file in test_cases.items():
        try:
            with test_file.open("r", encoding="utf-8") as f:
                file_content = f.read()
        except (UnicodeDecodeError, FileNotFoundError):
            continue

        try:
            tree = ast.parse(file_content, filename=str(test_file))
        except SyntaxError:
            continue

        # Identify the specific test function or method
        parts = test_id.split("::")
        if len(parts) == 2:
            # Standalone test function
            test_func_name = parts[1]
            target_node = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == test_func_name:
                    target_node = node
                    break
            if target_node:
                called_functions = get_called_functions(target_node, functions_to_implement)
                mapping[test_id].update(called_functions)
        elif len(parts) == 3:
            # Test method within a class
            class_name, test_func_name = parts[1], parts[2]
            target_node = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef) and child.name == test_func_name:
                            target_node = child
                            break
            if target_node:
                called_functions = get_called_functions(target_node, functions_to_implement)
                mapping[test_id].update(called_functions)

    return mapping


def get_called_functions(test_node: ast.FunctionDef, functions_to_implement: Set[str]) -> Set[str]:
    """
    Retrieves the set of functions that are called within a test function.

    Args:
        test_node (ast.FunctionDef): The AST node of the test function.
        functions_to_implement (Set[str]): Set of function/method names that need implementation.

    Returns:
        Set[str]: Set of function names that are called within the test function.
    """
    called_functions = set()

    for node in ast.walk(test_node):
        if isinstance(node, ast.Call):
            # Handle different types of function calls
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in functions_to_implement:
                    called_functions.add(func_name)
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
                if func_name in functions_to_implement:
                    called_functions.add(func_name)
    return called_functions


def display_mapping(mapping: Dict[str, Set[str]], library_name: str) -> None:
    """
    Displays the test to functions mapping using Rich tables.

    Args:
        mapping (Dict[str, Set[str]]): The mapping data.
        library_name (str): Name of the library.
    """
    if not mapping:
        console.print(f"[yellow]No mappings found for {library_name}.[/yellow]")
        return

    library_color = "blue"
    test_color = "magenta"
    function_color = "cyan"
    border_color = "green"

    table = Table(
        title=f"Library: [bold {library_color}]{library_name}[/bold {library_color}]",
        box=box.ROUNDED,
        show_header=True,
        header_style=f"bold {test_color}"
    )
    table.add_column("Test Case", style=test_color, overflow="fold")
    table.add_column("Functions Invoked", style=function_color, overflow="fold")

    for test_case, functions in sorted(mapping.items()):
        if functions:
            functions_str = ', '.join(sorted(functions))
        else:
            functions_str = "[red]No functions detected[/red]"
        table.add_row(test_case, functions_str)

    panel = Panel(
        table,
        border_style=border_color,
        padding=(1, 2)
    )
    console.print(panel)


def get_test_mapping(repo_path: Path,) -> Dict[str, Set[str]]:
  """
  Function which returns mapping of tests to their files, given path to the repository.
  """
  repo_name = repo_path.name
  source_files = find_source_files(repo_path)
  functions_to_implement = extract_functions_to_implement(source_files)
  if not functions_to_implement:
    print(f"No functions needing implementation found in {repo_name}")
    return None
      
  # Extract test files
  test_files = find_test_files(repo_path)
  if not test_files:
    print(f"No test files found in {repo_name}.")
    return None
      
  # Extract test cases
  test_cases = extract_test_cases(test_files)
  if not test_cases:
    print(f"No test cases found in {repo_name}.")
    return None

  # Map tests to functions
  mapping = map_tests_to_functions(test_cases, functions_to_implement)
  if not mapping:
    print(f"No mappings found for {repo_name}.")
    return None

  return mapping
        


def main():
    base_dir = Path('.').resolve()

    # Step 1: Identify libraries
    console.print("[bold blue]Identifying libraries...[/bold blue]")
    libraries = find_libraries(base_dir, recursive=True)
    if not libraries:
        console.print("[red]No libraries found in the current directory.[/red]")
        sys.exit(1)

    for lib in libraries:
        library_name = lib.name
        console.print(f"[bold blue]Processing Library: {library_name}[/bold blue]")

        # Step 2: Identify functions to implement
        source_files = find_source_files(lib)
        functions_to_implement = extract_functions_to_implement(source_files)
        if not functions_to_implement:
            console.print(f"[yellow]No functions needing implementation found in {library_name}.[/yellow]")
            continue

        # Step 3: Extract test files
        test_files = find_test_files(lib)
        if not test_files:
            console.print(f"[yellow]No test files found in {library_name}.[/yellow]")
            continue

        # Step 4: Extract test cases
        test_cases = extract_test_cases(test_files)
        if not test_cases:
            console.print(f"[yellow]No test cases found in {library_name}.[/yellow]")
            continue

        # Step 5: Map tests to functions
        mapping = map_tests_to_functions(test_cases, functions_to_implement)
        if not mapping:
            console.print(f"[yellow]No mappings found for {library_name}.[/yellow]")
            continue

        # Step 6: Display the mapping
        display_mapping(mapping, library_name)


if __name__ == "__main__":
    main()