from setuptools import setup
import pathlib
import sys

home_page = "https://github.com/sun-asterisk-research/table_reconstruction"
assert sys.version_info >= (
    3, 6, 2), "table_reconstruction requires Python 3.6.2+"

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="table_reconstruction",
    use_scm_version=True,
    author="Sun* AI Research",
    author_email="sun.converter.team@gmail.com",
    setup_requires=['setuptools_scm'],
    description="A table reconstruction package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=home_page,
    project_urls={
        "Bug Tracker": "{}/issues".format(home_page),
    },
    python_requires=">=3.6",
)
