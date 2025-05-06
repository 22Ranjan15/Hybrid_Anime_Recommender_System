import os 
from pathlib import Path

list_of_files = [
    "artifacts/.gitkeep",
    "config/__init__.py",
    "notebooks/experiments.ipynb",
    "pipeline/__init__.py",
    "pipeline/train_pipeline.py",
    "src/__init__.py",
    "src/components/__init__.py",
    "src/components/data_ingestion.py",
    "src/components/data_processer.py",
    "src/components/model_trainer.py",
    "src/logger.py",
    "src/exception.py",
    "static/.gitkeep",
    "templates/.gitkeep",
    "utils/__init__.py",
    "utils/utils.py",
    ".gitignore",
    "app.py",
    "README.md",
    "requirements.txt",
    "setup.py",
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)

    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, "w"):
            pass
    
# This code creates a project structure with the specified files and directories. 
