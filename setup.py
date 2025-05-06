from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    # This function will return the list of requirements
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="anime_recommender_system",
    version="0.0.1",
    author="Ranjan",
    author_email="ranjandasbd22@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)

""" 
    This code is a setup script for a Python package named "anime recommender system". It uses the setuptools 
    library to define the package's metadata and dependencies. The setup.py file is essential for distributing 
    the package, allowing it to be installed using pip or other package managers.
"""