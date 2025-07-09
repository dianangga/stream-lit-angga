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

# === Inisialisasi tools ===
stopword = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

def preprocess(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    print(f"🔍 Preprocessing: {text}...")  # Log first 30 chars
    text = re.sub(r'[^a-z\s]', '', text)
    text = stopword.remove(text)
    text = stemmer.stem(text)
    print(f"🔍 Preprocessing: {text}...")  # Log first 30 chars
    return text

# === Tabs ===
tab1, tab2 = st.tabs(["📌 Proses Lexicon + TF-IDF + SVM", "📌 Train Manual dari Label & Tweet"])

# ======================================================
# ===================== TAB 1 ==========================
# ======================================================
with tab1:
    st.title("📌 Tab 1: Proses Otomatis (Lexicon + TF-IDF + SVM)")

    ulasan_file = st.file_uploader("📄 Upload file 'ulasan.csv'", type=["csv"])
    lexicon_file = st.file_uploader("📚 Upload file 'lexicon.csv' (kata, label)", type=["csv"])

    if ulasan_file and lexicon_file:
        try:
            # === Load file
            df = pd.read_csv(ulasan_file)
            lexicon_df = pd.read_csv(lexicon_file)

            if 'ulasan' not in df.columns or 'kata' not in lexicon_df.columns or 'label' not in lexicon_df.columns:
                st.error("❌ Pastikan file memiliki kolom 'ulasan' dan lexicon punya 'kata', 'label'")
                st.stop()

            new_texts = df['ulasan'].astype(str).tolist()
            cleaned_texts = [preprocess(text) for text in new_texts]

            # === Load model, vectorizer, encoder ===
            with open('model/svm_model.pkl', 'rb') as f:
                model = pickle.load(f)
            with open('model/tfidf_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            with open('model/label_encoder.pkl', 'rb') as f:
                encoder = pickle.load(f)

            # === Lexicon Word Sets ===
            senang_words = set(lexicon_df[lexicon_df['label'].str.lower() == 'senang']['kata'].str.lower())
            marah_words = set(lexicon_df[lexicon_df['label'].str.lower() == 'marah']['kata'].str.lower())
            sedih_words = set(lexicon_df[lexicon_df['label'].str.lower() == 'sedih']['kata'].str.lower())
            netral_words = set(lexicon_df[lexicon_df['label'].str.lower() == 'netral']['kata'].str.lower())

            def label_by_lexicon(text):
                words = text.split()
                senang_match = any(w in senang_words for w in words)
                marah_match = any(w in marah_words for w in words)
                sedih_match = any(w in sedih_words for w in words)
                netral_match = any(w in netral_words for w in words)

                total_match = {
                    "senang": sum(w in senang_words for w in words),
                    "marah": sum(w in marah_words for w in words),
                    "sedih": sum(w in sedih_words for w in words),
                    "netral": sum(w in netral_words for w in words)
                }

                # Ambil label dengan jumlah match terbanyak
                if any(total_match.values()):
                    return max(total_match, key=total_match.get)
                else:
                    return "netral"
                  
            # === Proses Prediksi Gabungan ===
            results = []
            final_predictions = []

            for i, cleaned in enumerate(cleaned_texts):
              # SVM Prediction
              X_new = vectorizer.transform([cleaned])
              svm_pred = model.predict(X_new)
              svm_label = encoder.inverse_transform(svm_pred)[0]

              # Lexicon Prediction
              lexicon_label = label_by_lexicon(cleaned)

              # Final decision rule
              if svm_label == 'netral' or svm_label != lexicon_label:
                  final_label = lexicon_label
              else:
                  final_label = svm_label
                  
              final_predictions.append(final_label)

              results.append({
                  "Teks Asli": new_texts[i],
                  "Preprocessed": cleaned,
                  "Prediksi SVM": svm_label,
                  "Prediksi Lexicon": lexicon_label,
                  "Final Decision": final_label
              })

            st.subheader("📊 Hasil Prediksi Gabungan (SVM + Lexicon)")
            st.dataframe(pd.DataFrame(results))
            
            # === Confusion Matrix & Classification Report
            y_true = [label_by_lexicon(text) for text in cleaned_texts]
            y_pred = final_predictions

            # Ambil label dari lexicon + tambahan netral, sedih
            base_labels = lexicon_df['label'].str.lower().unique().tolist()
            mapped_labels = []
            if 'positif' in base_labels:
                mapped_labels.append('senang')
            if 'negatif' in base_labels:
                mapped_labels.append('marah')
            mapped_labels.extend(['netral', 'sedih'])
            mapped_labels = list(dict.fromkeys(mapped_labels))  # remove duplicates, preserve order

            st.subheader("📋 Classification Report")
            st.text(classification_report(
                y_true,
                y_pred,
                labels=mapped_labels,
                target_names=mapped_labels,
                digits=3
            ))

            st.subheader("🔍 Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred, labels=mapped_labels)
            cm_df = pd.DataFrame(cm,
                index=[f"Actual: {label.capitalize()}" for label in mapped_labels],
                columns=[f"Pred: {label.capitalize()}" for label in mapped_labels])
            st.dataframe(cm_df)


            # # === Download button
            # csv = pred_df.to_csv(index=False).encode('utf-8')
            # st.download_button("⬇️ Download Hasil Prediksi", csv, "hasil_prediksi.csv", "text/csv")


        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
# ======================================================
# ===================== TAB 2 ==========================
# ======================================================
with tab2:
    st.title("📌 Tab 2: Train Manual dari Dataset Label & Tweet")

    train_file = st.file_uploader("📄 Upload file CSV dengan kolom 'Label' dan 'Tweet'", type=["csv"])

    if train_file:
        try:
            df = pd.read_csv(train_file)

            if 'Tweet' not in df.columns or 'Label' not in df.columns:
                st.error("❌ Dataset harus memiliki kolom 'Tweet' dan 'Label'")
                st.stop()

            df['Tweet_cleaned'] = df['Tweet'].astype(str).apply(preprocess)

            encoder = LabelEncoder()
            df['label_encoded'] = encoder.fit_transform(df['Label'])

            X = df['Tweet_cleaned']
            y = df['label_encoded']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            vectorizer = TfidfVectorizer(min_df=1, max_df=0.95)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            model = SVC(kernel='linear', class_weight='balanced')
            model.fit(X_train_vec, y_train)

            y_pred = model.predict(X_test_vec)

            st.subheader("📊 Classification Report")
            report = classification_report(
                y_test,
                y_pred,
                labels=encoder.transform(encoder.classes_),
                target_names=encoder.classes_,
                output_dict=True
            )
            st.dataframe(pd.DataFrame(report).transpose())

            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
            st.subheader("📉 Confusion Matrix")
            st.dataframe(cm_df)

            # === Save Model, Vectorizer, Label Encoder ===
            os.makedirs('model', exist_ok=True)
            with open('model/svm_model.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('model/tfidf_vectorizer.pkl', 'wb') as f:
                pickle.dump(vectorizer, f)
            with open('model/label_encoder.pkl', 'wb') as f:
                pickle.dump(encoder, f)

            st.success("✅ Model dari Tab 2 berhasil disimpan di folder `model/`.")

        except Exception as e:
            st.error(f"❌ Gagal memproses: {e}")
