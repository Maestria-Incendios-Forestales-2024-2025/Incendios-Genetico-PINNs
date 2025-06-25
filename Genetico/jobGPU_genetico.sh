#! /bin/bash

#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q gpu
#$ -l gpu=1
#$ -l memoria_a_usar=1G
#$ -N genetic_alg

echo DeviceID: $SGE_GPU

#ejecutar el script
python algoritmo_genetico.py