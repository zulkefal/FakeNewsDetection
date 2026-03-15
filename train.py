import random
import numpy as np
from preprocess import load_data
from model import build_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)

X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer, model = build_model()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

param_grid = {
    "C": [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="f1",
    verbose=1
)

grid.fit(X_train_vec, y_train)

best_model = grid.best_estimator_

print("Best parameter:", grid.best_params_)

preds = best_model.predict(X_test_vec)

print(classification_report(y_test, preds))

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fake", "Real"],
            yticklabels=["Fake", "Real"])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# plt.savefig("confusion_matrix.png")
plt.show()