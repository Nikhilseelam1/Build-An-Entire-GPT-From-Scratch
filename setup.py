from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mini-gpt",
    version="0.1.0",
    author="Nikhil Seelam",
    description="A from-scratch implementation of a Mini GPT language model using PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    python_requires=">=3.8",
)
