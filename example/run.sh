#!/bin/bash
python_path="python"
inference_py="../inference.py"

$python_path $inference_py -r ./prot.pdb -l ./ligs.sdf -o result.tsv --model_path '../save'
