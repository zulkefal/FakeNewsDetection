# Fake News Detection

A Python project to classify news articles as **Fake** or **Real** using **TF-IDF features** and **Logistic Regression**. Includes data preprocessing, hyperparameter tuning, evaluation metrics, confusion matrix visualization, and reproducibility controls.

---

## Project Structure

```
FakeNewsDetection/
│
├── data/                # Raw dataset (not included in repo)
├── models/              # Saved model & vectorizer
├── preprocess.py        # Data loading & cleaning
├── model.py             # TF-IDF + Logistic Regression definition
├── train.py             # Training, evaluation, and saving pipeline
├── requirements.txt     # Python dependencies
├── tests/               # Unit tests
│   └── test_preprocess.py
├── README.md
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/zulkefal/fake-news-detection-ml.git
cd FakeNewsDetection
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## Usage

### Train the model

```bash
python train.py
```

This will:

- Preprocess the data  
- Train Logistic Regression with TF-IDF  
- Perform GridSearchCV hyperparameter tuning  
- Save the trained model and vectorizer to `models/`  
- Display classification report and confusion matrix  

### Example Results

- **Accuracy:** 0.99  
- **F1-score (weighted):** 0.99  
- **Confusion matrix** is saved as `confusion_matrix.png`.

---

## Reproducibility

- Random seeds set with `random.seed(42)` and `np.random.seed(42)`  
- GridSearchCV ensures consistent hyperparameter selection  
- Model and vectorizer saved for future inference

---

## Testing

Unit test for data loading: `tests/test_preprocess.py`  
Checks that `X` and `y` are non-empty and of equal length  

Run tests:

```bash
python -m pytest tests/
```

---

## Dataset

- Kaggle Fake and Real News Dataset (~44,000 articles)  
- Split: 80% train / 20% test  
- Cleaned: removed missing text entries, shuffled dataset

---

## Using the Saved Model (Optional)

```python
import pickle

# Load model
with open("models/logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Predict on new data
new_texts = ["Breaking news example..."]
X_new_vec = vectorizer.transform(new_texts)
preds = model.predict(X_new_vec)
print(preds)  # 0 = Fake, 1 = Real
```

---

## License

MIT License – compatible with open-source dataset and Python libraries

