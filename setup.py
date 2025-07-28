"""
Setup configuration for the File Processing Optimization system.
Implements requirement 1.1: Modular architecture with proper dependency management.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="file-processing-optimization",
    version="1.0.0",
    author="File Processing Team",
    author_email="team@fileprocessing.com",
    description="Optimized file processing and data matching system with fuzzy matching for Uzbek text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/company/file-processing-optimization",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: General",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "web": [
            "flask>=2.3.0",
            "flask-cors>=4.0.0",
            "gunicorn>=21.0.0",
        ],
        "performance": [
            "numba>=0.57.0",
            "cython>=3.0.0",
        ],
        "gpu": [
            "cupy-cuda11x>=12.0.0",  # For CUDA 11.x
            "cudf>=23.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "file-processor=application.cli:main",
            "file-processor-web=application.web:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
    },
    zip_safe=False,
)