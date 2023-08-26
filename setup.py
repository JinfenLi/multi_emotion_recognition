#!/usr/bin/env python
import os

from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()
with open(os.path.join("src", "version.txt")) as f:
    version = f.read().strip()

setup(
    name="multi_emotion_recognition",
    version=version,

    description="detect multiple emotions in a sentence including [anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness,\
                    surprise, trust]",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jinfen Li",
    author_email="jli284@syr.edu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    url="https://github.com/JinfenLi/multi_emotion_recognition",
    license="MIT",
    install_requires=["pytorch-lightning", "hydra-core"],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "command = main",
        ]
    },
)


import pytorch_lightning.trainer.trainer