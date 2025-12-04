#!/usr/bin/env python3
"""
Setup script for BlackSmith's Bookstore Chatbot
Creates a portable distribution package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="blacksmiths-bookstore-chatbot",
    version="1.0.0",
    author="James Deal",
    description="An intelligent bookstore chatbot with NLU capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jamesdeal89/HAICW",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.6.0",
    ],
    entry_points={
        "console_scripts": [
            "bookstore-chatbot=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "*.csv",
            "*.json",
            "*.md",
        ],
    },
)
