from setuptools import find_packages, setup

setup(
    name="example",
    packages=find_packages(exclude=["example_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
