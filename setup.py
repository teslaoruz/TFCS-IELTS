from setuptools import find_packages, setup


setup(
    name="tfcs-ielts",
    version="1.0.0",
    description="Three-tier cascaded scoring system for offline IELTS Writing assessment.",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.11",
)
