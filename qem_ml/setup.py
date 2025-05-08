# setup.py
from setuptools import setup, find_packages

setup(
    name="qenn",                     # Package name
    version="0.1.0",                 # Initial version
    author="Your Name",
    author_email="you@example.com",
    description="Quantum Error Neural Network package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/qenn_project",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=[
        "qiskit>=0.30.0",
        "tensorflow>=2.0",
        "numpy",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'qenn-train=training.train:main',
            'qenn-evaluate=evaluation.evaluate:main',
        ],
    },
)