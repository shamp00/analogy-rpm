#!/bin/bash
apt-get update && apt-get install -y --no-install-recommends \
    libcairo2-dev

pip install pycairo
pip install dataclasses
pip install numba
pip install colorama
pip install git+https://github.com/shamp00/pyRavenMatrices#egg=pyRavenMatrices

python RPM.py 