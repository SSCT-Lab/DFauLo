FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime


RUN /bin/bash -c "pip install opencv-python"
RUN /bin/bash -c "pip install tensorflow-gpu==2.8.0"
RUN /bin/bash -c "pip install matplotlib"
RUN /bin/bash -c "pip install pyod==2.0.2"
RUN /bin/bash -c "pip install torchtext==0.16.0"
RUN /bin/bash -c "pip install tqdm"
RUN /bin/bash -c "pip install cleanlab==2.0.0"
RUN /bin/bash -c "pip install wavemix==0.2.4"
RUN /bin/bash -c "pip install numpy==1.26.0"
RUN /bin/bash -c "pip install einops"
RUN /bin/bash -c "pip install pyyaml"
RUN /bin/bash -c "pip install deepod==0.2.0"
RUN /bin/bash -c "apt-get update"
RUN /bin/bash -c "apt-get install libgl1-mesa-glx"
RUN /bin/bash -c "apt-get update"
#可能需要选择时区
RUN /bin/bash -c "apt-get install libglib2.0-0" 


COPY . /workspace/dfaulo

WORKDIR /workspace/dfaulo

RUN /bin/bash -c "python exp_effective.py"

# 以上依赖可能不全，请查漏补缺：
# Package                      Version
# ---------------------------- -------------------
# absl-py                      2.1.0
# asttokens                    2.0.5
# astunparse                   1.6.3
# attrs                        23.1.0
# backcall                     0.2.0
# beautifulsoup4               4.12.2
# boltons                      23.0.0
# brotlipy                     0.7.0
# cachetools                   5.5.0
# certifi                      2023.7.22
# cffi                         1.15.1
# chardet                      4.0.0
# charset-normalizer           2.0.4
# cleanlab                     2.0.0
# click                        8.0.4
# conda                        23.9.0
# conda-build                  3.27.0
# conda-content-trust          0.2.0
# conda_index                  0.3.0
# conda-libmamba-solver        23.7.0
# conda-package-handling       2.2.0
# conda_package_streaming      0.9.0
# contourpy                    1.3.0
# cryptography                 41.0.3
# cycler                       0.12.1
# decorator                    5.1.1
# deepod                       0.2.0
# dnspython                    2.4.2
# einops                       0.8.0
# exceptiongroup               1.0.4
# executing                    0.8.3
# expecttest                   0.1.6
# filelock                     3.9.0
# flatbuffers                  24.3.25
# fonttools                    4.53.1
# fsspec                       2023.9.2
# gast                         0.6.0
# gmpy2                        2.1.2
# google-auth                  2.34.0
# google-auth-oauthlib         0.4.6
# google-pasta                 0.2.0
# grpcio                       1.66.1
# h5py                         3.11.0
# hypothesis                   6.87.2
# idna                         3.4
# ipython                      8.15.0
# jedi                         0.18.1
# Jinja2                       3.1.2
# joblib                       1.4.2
# jsonpatch                    1.32
# jsonpointer                  2.1
# keras                        2.8.0
# Keras-Preprocessing          1.1.2
# kiwisolver                   1.4.7
# libarchive-c                 2.9
# libclang                     18.1.1
# libmambapy                   1.4.1
# llvmlite                     0.43.0
# Markdown                     3.7
# MarkupSafe                   2.1.1
# matplotlib                   3.9.2
# matplotlib-inline            0.1.6
# mkl-fft                      1.3.8
# mkl-random                   1.2.4
# mkl-service                  2.4.0
# more-itertools               8.12.0
# mpmath                       1.3.0
# networkx                     3.1
# numba                        0.60.0
# numpy                        1.26.0
# oauthlib                     3.2.2
# opencv-python                4.10.0.84
# opt-einsum                   3.3.0
# packaging                    23.1
# pandas                       2.2.2
# parso                        0.8.3
# pexpect                      4.8.0
# pickleshare                  0.7.5
# Pillow                       10.0.1
# pip                          23.2.1
# pkginfo                      1.9.6
# pluggy                       1.0.0
# prompt-toolkit               3.0.36
# protobuf                     5.28.1
# psutil                       5.9.0
# ptyprocess                   0.7.0
# pure-eval                    0.2.2
# pyasn1                       0.6.1
# pyasn1_modules               0.4.1
# pycosat                      0.6.6
# pycparser                    2.21
# Pygments                     2.15.1
# pyod                         2.0.2
# pyOpenSSL                    23.2.0
# pyparsing                    3.1.4
# PySocks                      1.7.1
# python-dateutil              2.9.0.post0
# python-etcd                  0.4.5
# pytz                         2023.3.post1
# PyWavelets                   1.7.0
# PyYAML                       6.0
# requests                     2.31.0
# requests-oauthlib            2.0.0
# rsa                          4.9
# ruamel.yaml                  0.17.21
# ruamel.yaml.clib             0.2.6
# scikit-learn                 1.5.2
# scipy                        1.14.1
# setuptools                   68.0.0
# six                          1.16.0
# sortedcontainers             2.4.0
# soupsieve                    2.5
# stack-data                   0.2.0
# sympy                        1.11.1
# tensorboard                  2.8.0
# tensorboard-data-server      0.6.1
# tensorboard-plugin-wit       1.8.1
# tensorflow-gpu               2.8.0
# tensorflow-io-gcs-filesystem 0.37.1
# termcolor                    2.4.0
# tf-estimator-nightly         2.8.0.dev2021122109
# threadpoolctl                3.5.0
# tomli                        2.0.1
# toolz                        0.12.0
# torch                        2.1.0
# torchaudio                   2.1.0
# torchdata                    0.7.0
# torchelastic                 0.2.2
# torchtext                    0.16.0
# torchvision                  0.16.0
# tqdm                         4.65.0
# traitlets                    5.7.1
# triton                       2.1.0
# truststore                   0.8.0
# types-dataclasses            0.6.6
# typing_extensions            4.7.1
# tzdata                       2024.1
# urllib3                      1.26.16
# wavemix                      0.2.4
# wcwidth                      0.2.5
# Werkzeug                     3.0.4
# wheel                        0.41.2
# wrapt                        1.16.0
# zstandard                    0.19.0'