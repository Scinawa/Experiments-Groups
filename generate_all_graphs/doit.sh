#!/bin/bash 

#PBS -l mem=10GB

for i in para2
do
   python3 /home/svu/cqtales/generate_ALL_graphs/generate_graphs.py 9 $i
done
