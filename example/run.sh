#!/bin/bash
python_path="python"
inference_py="../inference.py"

$python_path $inference_py -r ./1KLT.pdb -l ligands.sdf -o result.tsv --model_path '../save'
