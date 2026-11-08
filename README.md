# Breast Cancer Diagnosis: Classification Model Comparison

Machine Learning(S1-26_AIMLZG565) - Assignment 2

**Live app:** https://mlclassificationmodels-amatxbaqhwhkrqi2erklvb.streamlit.app

## a. Problem statement

Predict whether a breast tumour is **malignant** or **benign** from numeric measurements of cell nuclei taken from digitised fine-needle-aspirate images. This is a binary classification problem. Five classifiers are trained on the same dataset and compared on Accuracy, AUC, Precision, Recall, F1 and MCC. The best model can then be explored interactively in a Streamlit app.

## b. Dataset description

- **Source:** Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository (Data downloaded using https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip).
- **Files** The dataset is in 2 files wdbc.data and wdbc.names. wdbc.names has the description of the columns and wdbc.data only has the data and no column headers.
- **Instances:** 569 (212 malignant, 357 benign), so there is a mild class imbalance of about 37% / 63%.
- **Features:** 30 numeric features. These are the mean, standard error and "worst" value of 10 nucleus measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.
- **Target:** `Diagnosis` (0 = benign, 1 = malignant). Malignant is treated as the positive class when computing Precision, Recall and F1, so Recall is the share of malignant tumours that are caught.
- **Missing values:** none.
- **Split:** 80/20 random train/test split with `random_state=42` (455 train / 114 test; 71 benign and 43 malignant in the test set). The test split is saved as `test_data.csv`.
- **Preprocessing:** `StandardScaler` inside the pipeline for Logistic Regression, KNN and SVM, so each saved model file carries its own scaling. Decision Tree and Gaussian Naive Bayes use raw features.

## c. GitHub repository link
`https://github.com/tejasviadiga/ml_classification_models/tree/main`

```
.
├── app.py                          # Streamlit app
├── ML_Assignment_2.ipynb           # training + evaluation notebook
├── requirements.txt
├── README.md
├── test_data.csv                   # held-out test set (features + target)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── metrics_summary.csv
```

### Python Version Used :
Python 3.13

### Run locally:

```bash
pip3 install -r requirements.txt
streamlit run app.py
```

Then upload `test_data.csv` in the sidebar. To regenerate the models and test data, run `ML_Assignment_2.ipynb` top to bottom.

## d. Models used

Evaluated on the 114-row held-out test set:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9737 | 0.9974 | 0.9762 | 0.9535 | 0.9647 | 0.9439 |
| Decision Tree | 0.9474 | 0.9355 | 0.9302 | 0.9302 | 0.9302 | 0.8880 |
| KNN | 0.9474 | 0.9820 | 0.9302 | 0.9302 | 0.9302 | 0.8880 |
| Naive Bayes | 0.9737 | 0.9984 | 1.0000 | 0.9302 | 0.9639 | 0.9447 |
| Support Vector Machine | 0.9561 | 0.9974 | 0.9318 | 0.9535 | 0.9425 | 0.9072 |

Hyperparameters: Logistic Regression (`max_iter=1000`), Decision Tree (`max_depth=5`), KNN (`k=5`), Gaussian Naive Bayes (defaults), SVM (RBF kernel, probabilities via `CalibratedClassifierCV(..., ensemble=False)`).

### Observations

Adding observations on the performance of each model on the chosen dataset. 

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Joint best accuracy (0.9737) with Naive Bayes, and the best F1 (0.9647). AUC is 0.9974 and MCC 0.9439. The classes are close to linearly separable once the features are scaled, so a simple linear model works very well. It is also fast and easy to interpret. |
| Decision Tree | Weakest model on AUC (0.9355) and MCC (0.8880), tied with KNN on accuracy (0.9474) and MCC. A single depth-5 tree has high variance, and its hard leaf probabilities give a low AUC. |
| KNN | Accuracy 0.9474 and MCC 0.8880, the same as the Decision Tree, but a much higher AUC (0.9820). It depends on feature scaling, and it is slower at prediction time. |
| Naive Bayes | Joint best accuracy (0.9737), the highest AUC (0.9984) and MCC (0.9447), and perfect precision (1.0000), so it never flags a benign tumour as malignant. Its recall is lower (0.9302), so it misses 3 of the 43 malignant tumours in the test set, which is the costlier error in diagnosis. The independence assumption is violated because many features are strongly correlated (radius, perimeter and area), yet it still ranks cases very well. |
| Support Vector Machine | Accuracy 0.9561, with the same AUC as Logistic Regression (0.9974) and the same recall (0.9535), but lower precision (0.9318) and MCC (0.9072). The RBF kernel gives no advantage over logistic regression here and is more costly to train. |
| **Overall winner** | **Logistic Regression** and **Naive Bayes** are effectively tied (accuracy 0.9737, MCC 0.9439 vs 0.9447). Logistic Regression has the better balance of precision and recall (best F1) and misses fewer malignant tumours (2 vs 3 of 43), which matters most in diagnosis, so it is preferred, along with its simplicity and interpretability. The test set is small (114 rows), so differences of one or two samples between models are not statistically meaningful. |

## Streamlit app features

- CSV upload (test data only)
- Model selection dropdown (5 models)
- Metrics display: Accuracy, AUC, Precision, Recall, F1, MCC
- Confusion matrix and classification report
- Side-by-side comparison of all models on the uploaded data
- Uses the saved models and does not retrain
