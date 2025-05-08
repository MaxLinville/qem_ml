# setup.py
from setuptools import setup, find_packages

setup(
    name="qem_ml",
    version="0.1.0",
    author="Max Linville",
    author_email="maxlinville7@gmail.com",
    description="Quantum Error Mitigation ML package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MaxLinville/qem_ml",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=[
        "qiskit",
        "qiskit_aer",
        "qiskit_ibm_runtime",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "joblib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)