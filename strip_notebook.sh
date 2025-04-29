#!/bin/bash
# filepath: d:\Vasco\UN\3.ano\2 semestre\IA\Projeto_2\strip_notebook.sh

# Strip outputs using nbconvert
python -m jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True --output loan_prediction_strict.ipynb loan_prediction.ipynb
echo "Notebook stripped and saved as loan_prediction_strict.ipynb"
