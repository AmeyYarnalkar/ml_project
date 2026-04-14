from setuptools import find_packages, setup


def get_requirements(file_path):
    """
    The function `get_requirements` reads a file and returns a list of requirements, excluding the "-e
    ." entry if present.
    
    :param 
    file_path: The function `get_requirements(file_path)` reads a file located at the specified
    `file_path` and returns a list of requirements. It removes the newline characters from each line and
    excludes the requirement "-e ." if it exists in the file
    :return: The function `get_requirements(file_path)` returns a list of requirements read from the
    file specified by the `file_path`. The function reads the file, removes newline characters from each
    line, and removes the requirement "-e ." if it exists in the list before returning the final list of
    requirements.
    """
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements


setup(
    name="ml_project",
    version="0.0.1",
    author="Amey Yarnalkar",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)