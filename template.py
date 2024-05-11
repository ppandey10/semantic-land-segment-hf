import os 
import logging 
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

project_name = 'landcover_segmentation'

files_list = [
    '.github/workflows/.gitkeep',
    f'src/{project_name}/__init__.py',
    f'src/{project_name}/components/__init__.py',
    f'src/{project_name}/logging/__init__.py',
    f'src/{project_name}/utils/__init__.py',
    f'src/{project_name}/utils/common.py',
    f'src/{project_name}/config/__init__.py',
    f'src/{project_name}/config/configuration.py',
    f'src/{project_name}/pipeline/__init__.py',
    f'src/{project_name}/entity/__init__.py',
    f'src/{project_name}/constants/__init__.py',
    'config/config.yaml',
    'params.yaml',
    'app.py',
    'main.py',
    'Docker',
    'requirements.txt',
    'setup.py',
    'experiments/test.ipynb'
]

for filepath in files_list:
    filepath = Path(filepath)
    file_dir, file_name = os.path.split(filepath)

    # create directory
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f'creating directory: {file_dir} for file: {file_name}')

    # create file
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f'creating empty file: {file_name}')

    else:
        logging.info(f'{file_name} already exists!')        
