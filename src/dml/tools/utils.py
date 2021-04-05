from pathlib import Path


def get_project_root():
    cwd = Path.cwd()
    root_idx = cwd.parts.index('dml')
    root_directory = Path(*cwd.parts[:root_idx + 1])
    return root_directory


def get_path_to_file(filename):
    root = get_project_root()
    file_path = list(root.rglob(filename))

    if len(file_path) == 0:
        return 0
    else:
        return file_path[0]

def generate_data(k=20, n=100):
    pass