import os
import time

TEMP_PATH = os.path.join("visualization", "temp") # Path for temporary file storage

def clear_temp() -> None:
    """Clears the temp directory of png files used in the visualizer"""
    files = os.listdir(TEMP_PATH)

    for file_name in files:
        file_path = os.path.join(TEMP_PATH, file_name)
        try:
            os.remove(file_path)
        except Exception:
            print(f"could not delete file {file_name}")

def make_tempfile_path() -> str:
    """Canonical temp path for png files"""
    return os.path.join(TEMP_PATH, f"TEMP{time.time_ns()}")