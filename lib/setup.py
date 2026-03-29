"""Triality - Automatic PDE Solver with Spatial Flow Engine"""

from setuptools import setup, find_packages

setup(
    name="triality",
    version="0.2.0",
    description="Real-time physics reasoning engine for rapid engineering analysis and decision-making",
    long_description=open("../README.md", encoding="utf-8").read() if __import__("os").path.exists("../README.md") else "",
    long_description_content_type="text/markdown",
    author="Genovation Technological Solutions Pvt Ltd",
    author_email="connect@genovationsolutions.com",
    url="https://github.com/genovation-tech/triality",
    project_urls={
        "Homepage": "https://genovationsolutions.com",
        "Documentation": "https://github.com/genovation-tech/triality/tree/main/docs",
        "Source": "https://github.com/genovation-tech/triality",
        "Issues": "https://github.com/genovation-tech/triality/issues",
    },
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
    ],
    extras_require={
        "plot": ["matplotlib>=3.4"],
        "test": ["pytest>=7.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
