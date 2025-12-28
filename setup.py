from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="auto-feature-engine",
    version="0.1.0",
    author="Feature Engineering Team",
    description="Automated Feature Engineering Engine for tabular data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "featuretools>=1.0.0",
        "numpy>=1.21.0",
    ],
)
