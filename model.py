from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def build_model():

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    return vectorizer, model