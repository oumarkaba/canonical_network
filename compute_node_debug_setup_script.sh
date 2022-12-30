#!/bin/bash

module load python/3.7
module load cudatoolkit

source ~/venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8