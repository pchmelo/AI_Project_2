#!/bin/bash
# filepath: d:\Vasco\UN\3.ano\2 semestre\IA\Projeto_2\restore_notebook.sh

# Check if the stripped version exists
if [ -f "loan_prediction_strict.ipynb" ]; then
    # Copy the stripped version to recreate the original notebook
    cp loan_prediction_strict.ipynb loan_prediction.ipynb
    echo "Original notebook (loan_prediction.ipynb) recreated from the stripped version."
else
    echo "Error: loan_prediction_strict.ipynb not found. Please ensure it exists after the git pull."
fi