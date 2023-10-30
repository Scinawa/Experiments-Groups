#!/bin/bash 

#conda activate /hpctmp/quantumwuongo/envs/sage310/

for i in {47..80}
do
  echo "[Bashozzo]: Working with $i"
  mkdir -p /hpctmp/quantumwuongo/data/representations/$i
  #python3 irrep.py 2 $i
  conda run -p /hpctmp/quantumwuongo/envs/sage310/ python3 /home/svu/cqtales/generate_irreps/irrep.py 2 $i
done
