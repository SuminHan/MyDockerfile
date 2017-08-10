#!/bin/bash
#SBATCH --ntasks=5
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1

stress -c 8 -i 1 -m 1 -t 10s
hostname
