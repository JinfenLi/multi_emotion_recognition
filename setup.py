#!/usr/bin/env python
import os

from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()
with open(os.path.join("multi_emotion_recognition", "version.txt")) as f:
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
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = multi_emotion_recognition.train:main",
            "eval_command = multi_emotion_recognition.eval:main",
        ]
    },
)


