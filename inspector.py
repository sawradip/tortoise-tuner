import ast
import sys
from collections import defaultdict

def extract_classes_and_functions_from_module(node):
    classes = [n.name for n in node.body if isinstance(n, ast.ClassDef)]
    functions = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
    return classes, functions

def compare_files(filenames):
    all_classes = defaultdict(list)
    all_functions = defaultdict(list)

    for filename in filenames:
        with open(filename, "r") as file:
            node = ast.parse(file.read(), filename=filename)
            classes, functions = extract_classes_and_functions_from_module(node)

            for cls in classes:
                all_classes[cls].append(filename)

            for func in functions:
                all_functions[func].append(filename)

    return all_classes, all_functions

def print_common_items(all_classes, all_functions):
    print("Classes defined in more than one file:")
    for cls, files in all_classes.items():
        if len(files) > 1:
            print(f"{cls}: found in files {', '.join(files)}")

    print("\nFunctions defined in more than one file:")
    for func, files in all_functions.items():
        if len(files) > 1:
            print(f"{func}: found in files {', '.join(files)}")

if __name__ == "__main__":
    filenames = sys.argv[1:]  # expects file paths as command-line arguments
    all_classes, all_functions = compare_files(filenames)
    print_common_items(all_classes, all_functions)
