"""
Setup script for the Gemini Video Transcript Analysis MVP.
"""

from setuptools import setup, find_packages
import os

# Read version from version.py
version_dict = {}
with open(os.path.join("gemini_mvp", "version.py")) as f:
    exec(f.read(), version_dict)

# Read README.md for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gemini_mvp",
    version=version_dict["__version__"],
    author="YourName",
    author_email="your.email@example.com",
    description="Optimized MVP for processing video transcripts with Google's Gemini API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gemini_mvp",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "google-generativeai>=0.3.0",
        "sentence-transformers>=2.2.2",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "click>=8.1.3",
        "tqdm>=4.65.0",
        "colorama>=0.4.6",
        "requests>=2.28.2",
        "cachetools>=5.3.0",
        "diskcache>=5.6.1",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "ui": [
            "streamlit>=1.22.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gemini-mvp=gemini_mvp.cli:cli",
        ],
    },
)
