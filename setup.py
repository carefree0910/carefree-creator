from setuptools import setup, find_packages

VERSION = "0.2.0"
PACKAGE_NAME = "carefree-creator"

DESCRIPTION = "An AI-powered creator for everyone."
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    entry_points={"console_scripts": ["cfcreator = cfcreator.cli:main"]},
    install_requires=[
        "click>=8.1.3",
        "fastapi==0.88.0",
        "carefree-client>=0.1.9",
        "carefree-learn[cv_full]>=0.4.0",
    ],
    extras_require={
        "kafka": [
            "kafka-python",
            "redis[hiredis]",
            "cos-python-sdk-v5",
        ]
    },
    author="carefree0910",
    author_email="syameimaru.saki@gmail.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords="python carefree-learn PyTorch",
)
