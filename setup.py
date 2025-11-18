from setuptools import setup, find_packages

setup(
    name="slm_lora_assignment",
    version="0.1.0",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().strip().split("\n"),
)