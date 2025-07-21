import pandas as pd
import re
import os
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# === STEP 1: Load data ===
data = pd.read_csv("dataset/text,emotion.csv")
texts = data['text']
labels = data['emotion']

# === STEP 2: Preprocessing ===
stemmer = StemmerFactory().create_stemmer()
stop_remover = StopWordRemoverFactory().create_stop_word_remover()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = stop_remover.remove(text)
    text = stemmer.stem(text)
    return text

cleaned_texts = texts.apply(preprocess)

# === STEP 3: TF-IDF ===
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_texts)

# === STEP 4: Label Encoding ===
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# === STEP 5: Split data (Stratified) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

# === STEP 6: Training Multi Kernel ===
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
os.makedirs('model', exist_ok=True)

for kernel in kernels:
    print(f"\n⚙️ Training SVM dengan kernel: {kernel} ...")
    model = SVC(kernel=kernel, probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("📊 Classification Report:")
    print(classification_report(
        y_test, y_pred,
        labels=encoder.transform(encoder.classes_),
        target_names=encoder.classes_
    ))

    print("🧾 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # === Simpan Model, Vectorizer & Encoder ===
    with open(f'model/svm_model_{kernel}.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(f'model/tfidf_vectorizer_{kernel}.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(f'model/label_encoder_{kernel}.pkl', 'wb') as f:
        pickle.dump(encoder, f)

    print(f"✅ Model kernel '{kernel}' berhasil disimpan!\n")
