#!/usr/bin/env python
import os

from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="multi_emotion",
    version='0.1.13',

    description="detect multiple emotions in a sentence including [anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness,\
                    surprise, trust]",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jinfen Li",
    author_email="jli284@syr.edu",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3.9'
    ],
    url="https://github.com/JinfenLi/multi_emotion_recognition",
    license="MIT",
    install_requires=["lightning>=2","torch>=2",
                      "emotlib",
                      "numpy", "pandas", "rich", "torchmetrics>=1",
                      "tqdm",
                      "transformers==4.31.0"],
    packages=find_packages(),


)
