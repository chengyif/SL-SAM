from setuptools import setup, find_packages

setup(
    name="llama_catastrophic_forgetting",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"":"src"},
)