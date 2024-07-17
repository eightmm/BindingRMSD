#!/bin/bash
python_path="/home/sim/anaconda3/envs/cu12.1/bin/python"
inference_py="../inference.py"

$python_path $inference_py -r ./1KLT.pdb -l ligands.sdf -o result.csv --model_path ../save 
