import pathlib

def make_parentdir_if_not_exist(path: str):
    dirpath = pathlib.Path(path).parent
    if dirpath.exists() and dirpath.is_dir():
        return
    else:
        dirpath.mkdir(parents=True, exist_ok=True)


