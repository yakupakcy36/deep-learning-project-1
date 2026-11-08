# Breast Cancer Classification (Neural Network)

Small learning project. The notebook `dl_project_1.ipynb` trains a small Keras Sequential network (Dense layers).

Data: Breast cancer Wisconsin diagnostic dataset (load_breast_cancer in the notebook), binary classification malignant/benign. The data files are not included in this repo, so you need to download them yourself.

## Benchmark script

`src/benchmark.py` compares L2 logistic regression, a shallow MLP and a deeper MLP (64-32) with 5-fold cross-validation.
It uses scikit-learn's built-in breast cancer dataset (569 samples, 30 features).

## How to run

```
pip install numpy scikit-learn tensorflow jupyter
jupyter notebook
python src/benchmark.py
```


## Results (5-fold CV, mean)

| Model | roc_auc | accuracy | f1 |
|---|---|---|---|
| L2 Logistic Regression Baseline | 0.995 | 0.9754 | 0.9807 |
| Shallow MLP (Notebook 20-Unit) | 0.995 | 0.9754 | 0.9805 |
| Deep MLP (64-32 ReLU + L2) | 0.994 | 0.9736 | 0.9792 |

Full numbers are in `benchmarks/benchmark_results.json`.

## Files

- `dl_project_1.ipynb`: notebook
- `src/benchmark.py`: benchmark script
- `benchmarks/benchmark_results.json`: results
- `assets/benchmark_chart.svg`: chart of the results
