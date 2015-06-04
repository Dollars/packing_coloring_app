#!/bin/bash -l

#PBS -l walltime=6:00:00
#PBS -o test_tabu.out
#PBS -e test_tabu.err
#PBS -N test_tabu.py

module load graph-tool/2.2.42-foss-2015a-Python-2.7.9

# Go in the submission directory where the R script is located:
export PYTHONPATH=$PYTHONPATH:$HOME/packing_coloring_app/
cd $HOME/packing_coloring_app/benchmarks

python test_tabu.py
