from setuptools import setup, find_packages

setup(
    name="your_project_name",
    version="0.1.0",
    packages=find_packages(include=["traking", "traking.*"]),  # replace with your actual package
    package_dir={"": "."},  # flat layout: top-level folder is the root
    install_requires=[
        "some-lib",
        "another-lib>=1.0",
    ],
)

