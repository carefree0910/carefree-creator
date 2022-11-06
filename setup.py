from setuptools import setup, find_packages

VERSION = "0.1.0"
PACKAGE_NAME = "carefree-creator"

DESCRIPTION = "An AI-powered creator for everyone."
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "carefree-client>=0.1.6",
        "carefree-learn[cv]>=0.3.3.15",
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
