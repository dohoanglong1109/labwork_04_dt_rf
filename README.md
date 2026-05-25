# LABWORK REPORT: EVALUATING ENSEMBLE LEARNING VIA CUSTOM RANDOM FOREST
## 1. Project Overview
This project explores the empirical performance and generalization capabilities of two supervised learning algorithms: a single Baseline Decision Tree and a Custom Random Forest ensemble (**$K = 100$**). The primary objective is to analyze the effectiveness of Bootstrap Aggregation (Bagging) and Majority Voting in mitigating the inherent high-variance vulnerability of unpruned decision trees across two distinct data domains.

## 2. Dataset Descriptions
The project evaluates models on two contrasted datasets located in the `data/` directory:
- **Wisconsin Breast Cancer Dataset**: Consists of highly distinct continuous geometric features used to classify tumors as Benign or Malignant.
- **Wine Quality Dataset**: A more challenging dataset containing physiochemical properties of red wine. The target is binarized to evaluate wine quality scores (Good vs. Bad).

## 3. Repository Structure
The project is modularized into 5 core Python files and an automation script:
- `preprocessing.py`: Handles raw data loading, feature mapping, stratification, 80/20 train-test splitting, and exports clean data to `data/processed/`.
- `models.py`: A modular file containing the scratch implementation class for the `CustomRandomForest` ensemble.
- `baseline_dt.py`: Trains and evaluates the unpruned baseline Decision Tree to establish baseline performance metrics.
- `bagging_ensemble.py`: Simulates the Bootstrap Sampling process, calculates empirical Out-of-Bag (OOB) error rates.
- `evaluate_models.py`: Executable script that trains both models, compares their training/testing classification errors, and outputs the final comparative summary.
- `run_pipeline.sh`: A Bash automation script to execute the entire end-to-end machine learning pipeline with a single command.
- `.gitignore`: Restricts data caches (`data/processed/`), compiled python scripts (`__pycache__/`), and local figures from being tracked by Git.

## 4. Installation & Prerequisites
This project is developed on a Linux environment using **Miniconda** for package management.
**Required Packages:**
- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn

**Setup Environment:**
```
# Create and activate a new conda environment
conda create -n ml_env python=3.10 -y
conda activate ml_env

# Install required packages
conda install pandas numpy matplotlib scikit-learn -y
```

## 5. How to Run the Pipeline
To execute the data preprocessing, baseline training, bootstrapping analysis, and final model evaluation in a seamless sequential flow, run the automated shell script:
```
# Grant execution permission to the script
chmod +x run_pipeline.sh

# Run the full pipeline
./run_pipeline.sh
```

## 6. Summary of Empirical Results
**Baseline Hyperparameters (Table 3.1):**
- Criterion: "gini"
- Max Depth: None (Unpruned)
- Min Samples Split: 2
- Random State: 42

Comprehensive Testing Error Comparison (Table 5.1)
The models achieved the following Classification Errors (%) on the unseen testing subset:

| Dataset | Decision Tree Error (%) | Random Forest Error (%) | Relative Error Reduction (Δ) |
| :--- | :---: | :---: | :---: |
| Breast Cancer | 7.02% | 2.63% | 4.39% |
| Wine Quality | 39.38% | 33.44% | 5.94% |

## 7. Key Findings & Conclusion
- **Overfitting Proof**: The single unpruned Decision Tree achieved a 0.00% training error on both datasets, demonstrating high-variance behavior by memorizing data noise instead of learning generalized patterns.
- **Variance Reduction**: The Custom Random Forest successfully smoothed the decision boundaries through Bagging. By aggregating 100 uncorrelated trees, the ensemble filtered out local variations, significantly cutting down testing classification errors across both domains.
