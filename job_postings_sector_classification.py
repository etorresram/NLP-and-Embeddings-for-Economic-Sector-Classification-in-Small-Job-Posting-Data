import random
import numpy as np
import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('omw-1.4')
#!unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Set global seeds
random.seed(42)
np.random.seed(42)

# Load the dataset 
df_full = pd.read_csv('/kaggle/input/dataset-job-postings/jobs_data_5000.csv')

stop_en = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_en]
    words = [lemmatizer.lemmatize(w) for w in words]
    text = " ".join(words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_pipeline(df):
    df["clean_description"] = df["description_en"].apply(clean_text)
    le = LabelEncoder()
    df["naics_encoded"] = le.fit_transform(df["naics_code"].astype(str))
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_description"], df["naics_encoded"], test_size=0.2, random_state=42)
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    accuracy_results = {}
    
    # Models using TF-IDF
    tfidf_models = {
        "TF-IDF - Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "TF-IDF - Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "TF-IDF - SVM": SVC(kernel='linear', random_state=42),
        "TF-IDF - XGBoost": XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42)}
    
    for model_name, model in tfidf_models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        accuracy_results[model_name] = accuracy_score(y_test, y_pred)
    
    # ====== WORD2VEC ======
    word2vec_model = Word2Vec(
        sentences=[text.split() for text in X_train],
        vector_size=200, window=10, min_count=1, epochs=50, workers=1, seed=42
    )
    def word2vec_features(text, model, vector_size=200):
        words = text.split()
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(vector_size)
    X_train_w2v = np.array([word2vec_features(text, word2vec_model, 200) for text in X_train])
    X_test_w2v = np.array([word2vec_features(text, word2vec_model, 200) for text in X_test])
    
    w2v_models = {
        "Word2Vec + Logistic": LogisticRegression(max_iter=500, random_state=42),
        "Word2Vec + RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Word2Vec + XGBoost": XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42),
        "Word2Vec + SVM": SVC(kernel='linear', random_state=42)}
    
    for model_name, model in w2v_models.items():
        model.fit(X_train_w2v, y_train)
        y_pred = model.predict(X_test_w2v)
        accuracy_results[model_name] = accuracy_score(y_test, y_pred)
    
    # ====== DOC2VEC ======
    tagged_docs = [TaggedDocument(words=text.split(), tags=[i]) for i, text in enumerate(X_train)]
    doc2vec_model = Doc2Vec(
        tagged_docs, vector_size=200, window=10, min_count=1, epochs=50, workers=1, seed=42
    )
    def stable_doc2vec_inference(text, model, epochs=20, n_infer=3):
        vectors = np.array([model.infer_vector(text.split(), epochs=epochs) for _ in range(n_infer)])
        return np.mean(vectors, axis=0)
    X_train_d2v = np.array([stable_doc2vec_inference(text, doc2vec_model, epochs=20, n_infer=3) for text in X_train])
    X_test_d2v = np.array([stable_doc2vec_inference(text, doc2vec_model, epochs=20, n_infer=3) for text in X_test])
    
    d2v_models = {
        "Doc2Vec + Logistic": LogisticRegression(max_iter=500, random_state=42),
        "Doc2Vec + RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Doc2Vec + XGBoost": XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42),
        "Doc2Vec + SVM": SVC(kernel='linear', random_state=42)}
    
    for model_name, model in d2v_models.items():
        model.fit(X_train_d2v, y_train)
        y_pred = model.predict(X_test_d2v)
        accuracy_results[model_name] = accuracy_score(y_test, y_pred)
    
    return accuracy_results

# Run the pipeline for different sample sizes
sample_sizes = [1000, 2000, 5000]
results_dict = {}  

for n in sample_sizes:
    df_subset = df_full.head(n).copy()
    acc_results = run_pipeline(df_subset)
    for model_name, acc in acc_results.items():
        if model_name not in results_dict:
            results_dict[model_name] = {}
        results_dict[model_name][f"n={n}"] = acc

accuracy_df = pd.DataFrame(results_dict).T
accuracy_df = accuracy_df[[f"n={n}" for n in sample_sizes]] 
accuracy_df = (accuracy_df * 100).round(2).astype(str) + "%"

# Visualization
plt.figure(figsize=(8,6))
accuracy_df_float = accuracy_df.replace('%','', regex=True).astype(float)
sns.heatmap(accuracy_df_float, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Model Accuracy Comparison (%) for Different Sample Sizes")
plt.show()