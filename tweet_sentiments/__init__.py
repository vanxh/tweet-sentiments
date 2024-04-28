import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        data = data.dropna(subset=['text'])

        data['text'] = data['text'].str.replace(r"http\S+", "", regex=True)\
                                    .str.replace(r"@\w+", "", regex=True)\
                                    .str.replace(r"[^A-Za-z0-9\s]+", "", regex=True)\
                                    .str.lower()

        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def get_tfidf(data):
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    features = tfidf.fit_transform(data['text'])
    labels = data['sentiment'].factorize()[0]
    return features, labels, tfidf

def main():
    data = load_data("data.csv")
    features, labels, tfidf = get_tfidf(data)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    prediction = model.predict(tfidf.transform(["you are leaving me alone i am so happy"]))
    prediction_transformed = data['sentiment'].factorize()[1][prediction][0]
    print(prediction_transformed)

if __name__ == "__main__":
    main()
