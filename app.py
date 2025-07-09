import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import re

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

st.title("💬 Klasifikasi Sentimen Ulasan Toko (SVM + Lexicon + Preprocessing)")

# === Inisialisasi Preprocessing Tools ===
stopword = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text

# Upload dataset ulasan
ulasan_file = st.file_uploader("📄 Upload file CSV berisi kolom 'ulasan'", type=["csv"])

# Upload lexicon
lexicon_file = st.file_uploader("📚 Upload file lexicon (kolom: kata, label)", type=["csv"])

if ulasan_file and lexicon_file:
    try:
        df = pd.read_csv(ulasan_file)
        lexicon_df = pd.read_csv(lexicon_file)

        if 'ulasan' not in df.columns:
            st.error("❌ Kolom 'ulasan' tidak ditemukan.")
            st.stop()
        if 'kata' not in lexicon_df.columns or 'label' not in lexicon_df.columns:
            st.error("❌ Lexicon harus punya kolom 'kata' dan 'label'")
            st.stop()

        os.makedirs("dataset", exist_ok=True)
        df.to_csv("dataset/ulasan_scraped.csv", index=False)

        st.subheader("📃 Data Awal")
        st.dataframe(df.head())

        # Preprocessing
        df['ulasan_preprocessed'] = df['ulasan'].astype(str).apply(preprocess)

        st.subheader("🧼 Setelah Preprocessing")
        st.dataframe(df[['ulasan', 'ulasan_preprocessed']].head())

        # Lexicon
        positif_words = set(lexicon_df[lexicon_df['label'].str.lower() == 'positif']['kata'].str.lower())
        negatif_words = set(lexicon_df[lexicon_df['label'].str.lower() == 'negatif']['kata'].str.lower())

        def label_by_lexicon(text):
            words = text.split()
            pos_match = any(w in positif_words for w in words)
            neg_match = any(w in negatif_words for w in words)
            if pos_match and not neg_match:
                return "senang"
            elif neg_match and not pos_match:
                return "marah"
            elif pos_match and neg_match:
                return "netral"
            else:
                return "sedih"

        df['label_text'] = df['ulasan_preprocessed'].apply(label_by_lexicon)
        encoder = LabelEncoder()
        df['label'] = encoder.fit_transform(df['label_text'])

        st.subheader("📄 Dataset Setelah Diberi Label")
        st.dataframe(df[['ulasan', 'ulasan_preprocessed', 'label_text', 'label']].head())

        # TF-IDF & Train
        X = df['ulasan_preprocessed']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        vectorizer = TfidfVectorizer(min_df=1, max_df=0.95)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        if len(np.unique(y_train)) < 2:
            st.error("❌ Data hanya memiliki satu kelas.")
            st.stop()

        model = SVC(kernel='linear', class_weight='balanced')
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)

        st.subheader("📋 Classification Report")
        report = classification_report(
            y_test,
            y_pred,
            labels=encoder.transform(encoder.classes_),
            target_names=encoder.classes_,
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
        st.dataframe(cm_df)

        os.makedirs("model", exist_ok=True)
        with open("model/svm_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("model/tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        with open("model/label_encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)

        st.success("✅ Model, vectorizer, dan label encoder berhasil disimpan.")

        # === Prediksi CSV Baru ===
        st.subheader("🧪 Uji Prediksi dari File CSV Baru")

        predict_file = st.file_uploader("📄 Upload file CSV berisi kolom 'ulasan' untuk diprediksi", type=["csv"])

        if predict_file:
            df_new = pd.read_csv(predict_file)

            if 'ulasan' not in df_new.columns:
                st.error("❌ Kolom 'ulasan' tidak ditemukan.")
                st.stop()

            df_new['ulasan_preprocessed'] = df_new['ulasan'].astype(str).apply(preprocess)

            def lexicon_predict(text):
                words = text.split()
                pos_match = any(w in positif_words for w in words)
                neg_match = any(w in negatif_words for w in words)
                if pos_match and not neg_match:
                    return "senang"
                elif neg_match and not pos_match:
                    return "marah"
                elif pos_match and neg_match:
                    return "netral"
                else:
                    return "sedih"

            svm_preds = []
            lexicon_preds = []
            final_preds = []

            for cleaned in df_new['ulasan_preprocessed']:
                X_new = vectorizer.transform([cleaned])
                svm_pred = model.predict(X_new)
                svm_label = encoder.inverse_transform(svm_pred)[0]
                lex_label = lexicon_predict(cleaned)

                final_label = lex_label if (svm_label == 'netral' or svm_label != lex_label) else svm_label

                svm_preds.append(svm_label)
                lexicon_preds.append(lex_label)
                final_preds.append(final_label)

            df_new['prediksi_svm'] = svm_preds
            df_new['prediksi_lexicon'] = lexicon_preds
            df_new['final_label'] = final_preds

            st.subheader("📊 Hasil Prediksi")
            st.dataframe(df_new[['ulasan', 'ulasan_preprocessed', 'prediksi_svm', 'prediksi_lexicon', 'final_label']])

            # Tombol download
            csv_download = df_new.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Hasil Prediksi", data=csv_download, file_name="hasil_prediksi.csv", mime='text/csv')

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
