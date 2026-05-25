#!/bin/bash

echo "Starting Machine Learning Pipeline..."
python3 preprocessing.py
python3 baseline_dt.py
python3 bagging_ensemble.py
python3 evaluate_models.py
echo "Pipeline Completed!"
