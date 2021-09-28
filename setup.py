import pathlib
import sys

from setuptools import find_packages, setup

home_page = "https://github.com/sun-asterisk-research/table_reconstruction"
assert sys.version_info >= (3, 7, 0), "table_reconstruction requires Python 3.7+"

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="table_reconstruction",
    use_scm_version={
        "write_to": "table_reconstruction/__version__.py",
        "version_scheme": "guess-next-dev",
        "local_scheme": "no-local-version",
    },
    author="Sun* AI Research",
    author_email="sun.converter.team@gmail.com",
    setup_requires=["setuptools_scm"],
    description="A table reconstruction package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=home_page,
    project_urls={
        "Bug Tracker": "{}/issues".format(home_page),
    },
    include_package_data=True,
    install_requires=requirements,
    package_dir={"": "."},
    packages=find_packages(where=".", exclude=["tests"]),
    command_options={
        "build_sphinx": {
            "project": ("setup.py", "Table Reconstruction"),
            "source_dir": ("setup.py", "docs/source"),
        }
    },
    python_requires='>=3.7.0',
)
