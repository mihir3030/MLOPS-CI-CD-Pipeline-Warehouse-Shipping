from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
REPO_NAME = "MLOPS-CI-CD-Pipeline-Warehouse-Shipping"
AUTHOR_USER_NAME = "mihir3030"
SRC_REPO = "src"
LIST_OF_REQUIREMENT = []


setup(
    name = SRC_REPO,
    version = '0.0.1',
    author = AUTHOR_USER_NAME,
    description = "AIops-mlflow-ci-cd-pipeline",
    long_description=long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/mihir3030/MLOPS-CI-CD-Pipeline-Warehouse-Shipping.git",
    author_email = "mihirdholakia777@Gmail.com",
    packages = [SRC_REPO],
    license = "MIT",
    python_required = ">=3.6",
    install_required = LIST_OF_REQUIREMENT
)