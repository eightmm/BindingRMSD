#!/bin/bash

# BindingRMSD Example Inference Script
# This script demonstrates how to run RMSD prediction on example data

echo "🧬 Running BindingRMSD inference on example data..."
echo "📁 Input protein: prot.pdb"
echo "📁 Input ligands: ligs.sdf"
echo "📁 Output: binding_results.tsv"
echo

# Method 1: Using the installed command (recommended)
if command -v bindingrmsd-inference &> /dev/null; then
    echo "✅ Using installed bindingrmsd-inference command"
    bindingrmsd-inference \
        -r prot.pdb \
        -l ligs.sdf \
        -o binding_results.tsv \
        --model_path ../save \
        --batch_size 64 \
        --device cuda
else
    echo "⚠️  bindingrmsd-inference command not found, using Python module"
    # Method 2: Using Python module directly
    cd ..
    python -m bindingrmsd.inference \
        -r example/prot.pdb \
        -l example/ligs.sdf \
        -o example/binding_results.tsv \
        --model_path save \
        --batch_size 64 \
        --device cuda
    cd example
fi

echo
echo "✅ Inference completed! Check binding_results.tsv for results."
echo "📊 Preview of results:"
echo
head -5 binding_results.tsv 2>/dev/null || head -5 result.tsv
