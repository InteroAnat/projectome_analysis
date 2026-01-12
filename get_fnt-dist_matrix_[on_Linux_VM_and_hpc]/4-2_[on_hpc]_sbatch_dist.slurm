#!/bin/sh

#SBATCH -J test
#SBATCH --partition=grande
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH -o output.txt
#SBATCH -e error.txt
#SBATCH --time=72:00:00 

fnt-dist -o dist.txt fnt_decimate_join.fnt
