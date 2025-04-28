#!/bin/bash
  
#SBATCH -J SCRIPT_NAME
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH --mail-user=ninaf@utexas.edu
#SBATCH --mail-type=all
#SBATCH -A AST23034

source $HOME/.bashrc
source $HOME/miniconda3/etc/profile.d/conda.sh

conda init bash
conda activate $HOME/miniconda3/envs/py311

module unload python3

export QT_QPA_PLATFORM=offscreen

./script.py 1>script_1.out 2>script_1.err
