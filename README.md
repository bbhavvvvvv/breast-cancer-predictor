# breast-cancer-predictor


A machine learning model that classifies breast tumors as **malignant** or **benign** from 30 measurements of cell nuclei, using the Wisconsin Diagnostic Breast Cancer dataset (bundled with scikit-learn, so no download is needed).

> **Disclaimer:** This is an educational project. It is **not** a medical device and must not be used for real diagnosis.

## Results

Trained on 455 samples, tested on 114 held-out samples:

| Metric   | Score  |
|----------|--------|
| Accuracy | 98.25% |
| ROC AUC  | 0.995  |

The test set had 1 missed malignant case and 1 false alarm.

![Evaluation](evaluation.png)
![Feature importance](feature_importance.png)

## How it works

1. Loads the dataset and makes a stratified 80/20 train/test split.
2. Compares Logistic Regression, SVM (RBF), Random Forest, and Gradient Boosting using 5-fold cross-validation (ROC AUC).
3. Trains the best model on the training set. Scaling lives inside a pipeline, so there is no data leakage.
4. Evaluates on the test set and saves plots.
5. Saves the trained model to `breast_cancer_model.joblib`.

## Setup

```bash
git clone https://github.com/<your-username>/breast-cancer-predictor.git
cd breast-cancer-predictor
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Train, evaluate, and save the model:

```bash
python breast_cancer_predictor.py
```

Also run sample predictions:

```bash
python breast_cancer_predictor.py --demo
```

Predict from your own code (after training once):

```python
from breast_cancer_predictor import predict

# patient is a dict with all 30 feature columns
print(predict(patient))
# [{'prediction': 'Benign', 'probability_malignant': 0.0021, 'probability_benign': 0.9979}]
```

## Limitations

- Trained on one small, clean dataset (569 samples).
- Not validated on real clinical populations.
- Predictions come from tumor measurements only, not imaging or patient history.

## Web app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Adjust the 30 tumor measurements with sliders, load a typical benign or malignant example, or upload a CSV for batch predictions.

## License

MIT
