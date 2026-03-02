"""
Setup script for 3D U-Net Soma Detection package.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="soma-detection-unet",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="3D U-Net for automated soma detection and volume measurement",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/soma-detection-unet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "advanced": [
            "monai>=0.8.0",
            "wandb>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "soma-detection=soma_detection_unet.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "soma_detection_unet": ["configs/*.yaml"],
    },
)