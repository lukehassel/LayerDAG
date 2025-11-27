#!/bin/bash
# Load Python module and activate virtual environment

module load Python/3.12.3
source ../venv/bin/activate

echo "Python module loaded and virtual environment activated"
python3 --version
