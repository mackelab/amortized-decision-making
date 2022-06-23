from setuptools import find_packages, setup

# Package meta-data.
NAME = "loss-calibration"
URL = "https://github.com/mackelab/loss-calibration"
EMAIL = "mila.gorecki@student.uni-tuebingen.de"
AUTHOR = "Mila Gorecki"
REQUIRES_PYTHON = ">=3.6.0"

REQUIRED = ["torch", "numpy", "matplotlib", "sbi", "sbibm"]

setup(
    name=NAME,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED,
)
