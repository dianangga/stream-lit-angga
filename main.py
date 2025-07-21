import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import re
import time
from urllib.parse import urlparse

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import scraping.tokopedia_scraper as tp

# === Inisialisasi tools ===
stopword = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text

def tokenize(text):
    return text.split()

# === Login Section ===
st.sidebar.title("🔐 Login")
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    username_input = st.sidebar.text_input("Username")
    password_input = st.sidebar.text_input("Password", type="password")
    login_button = st.sidebar.button("Login")

    # Ganti sesuai kebutuhan
    valid_username = "admin"
    valid_password = "123456"

    if login_button:
        if username_input == valid_username and password_input == valid_password:
            st.session_state.logged_in = True
            st.success("✅ Login berhasil!")
            st.rerun(scope="app")
        else:
            st.error("❌ Username atau password salah.")
    st.stop()  # stop aplikasi jika belum login
else:
    st.sidebar.success("👋 Selamat datang!")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun(scope="app")

# === Tabs ===
tab1, tab2, tab3 = st.tabs(["📌 Proses Lexicon + TF-IDF + SVM", "📌 Scraping & Proses Lexicon + TF-IDF + SVM", "📌 Train Manual dari Label & Tweet"])

# ======================================================
# ===================== TAB 1 ==========================
# ======================================================
with tab1:
    st.title("📌 Tab 1: Proses Otomatis (Lexicon + TF-IDF + SVM)")

    ulasan_file = st.file_uploader("📄 Upload file 'ulasan.csv'", type=["csv"])

    if ulasan_file:
        try:
            # === Load file
            df = pd.read_csv(ulasan_file)
            lexicon_df = pd.read_csv("lexicon/lexicon.csv")


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
                total_match = {
                    "senang": sum(w in senang_words for w in words),
                    "marah": sum(w in marah_words for w in words),
                    "sedih": sum(w in sedih_words for w in words),
                    "netral": sum(w in netral_words for w in words)
                }
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

            base_labels = lexicon_df['label'].str.lower().unique().tolist()
            mapped_labels = []
            if 'positif' in base_labels:
                mapped_labels.append('senang')
            if 'negatif' in base_labels:
                mapped_labels.append('marah')
            mapped_labels.extend(['netral', 'sedih'])
            mapped_labels = list(dict.fromkeys(mapped_labels))  # remove duplicates

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

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
with tab2:
    st.title("📌 Tab 2: Proses Otomatis dari Scraping Tokopedia (Lexicon + TF-IDF + SVM)")

    url_input = st.text_input("🔗 Masukkan URL produk Tokopedia")

    kernel_option = st.selectbox(
        "🔧 Pilih Kernel SVM",
        options=["linear", "poly", "rbf", "sigmoid"],
        index=0
    )

    if url_input:
        scrape_button = st.button("🚀 Mulai Scraping & Analisis")

        if scrape_button:
            try:
                parsed_url = urlparse(url_input)

                if  parsed_url.hostname != "www.tokopedia.com":
                    raise Exception("URL bukan dari Tokopedia")

                st.info("📡 Membuka halaman Tokopedia...")
                driver = tp.get_driver()

                if tp.open_url(driver, url_input):
                    st.info("📥 Scraping komentar produk...")

                    comments = tp.scrape_tokopedia_reviews(driver)
                    driver.quit()

                    st.subheader("💬 Komentar yang berhasil di-scrape")
                    st.write(f"Total komentar ditemukan: {len(comments)}")
                    st.dataframe(pd.DataFrame(comments, columns=["Komentar"]))

                    # === Preprocessing
                    new_texts = comments
                    tokenized_texts = [tokenize(preprocess(text)) for text in new_texts]
                    cleaned_texts = [' '.join(tokens) for tokens in tokenized_texts]

                    # === Load model SVM sesuai kernel
                    try:
                        with open(f'model/svm_model_{kernel_option}.pkl', 'rb') as f:
                            model = pickle.load(f)
                        with open('model/tfidf_vectorizer.pkl', 'rb') as f:
                            vectorizer = pickle.load(f)
                        with open('model/label_encoder.pkl', 'rb') as f:
                            encoder = pickle.load(f)
                    except FileNotFoundError:
                        st.error(f"❌ Model untuk kernel '{kernel_option}' belum dilatih. Silakan latih terlebih dahulu di Tab 3 atau pastikan file 'svm_model_{kernel_option}.pkl' ada.")
                        st.stop()

                    # === Load Lexicon (senang, marah, sedih saja)
                    nrc_df = pd.read_csv("lexicon/Indonesian-NRC-EmoLex.csv", sep=";", encoding="utf-8")
                    senang_words = set(nrc_df[nrc_df['joy'] == 1]['Indonesian Word'].str.lower())
                    marah_words = set(nrc_df[nrc_df['anger'] == 1]['Indonesian Word'].str.lower())
                    sedih_words = set(nrc_df[nrc_df['sadness'] == 1]['Indonesian Word'].str.lower())

                    def label_by_lexicon(tokens):
                        total_match = {
                            "senang": sum(w in senang_words for w in tokens),
                            "marah": sum(w in marah_words for w in tokens),
                            "sedih": sum(w in sedih_words for w in tokens),
                        }
                        values = list(total_match.values())

                        # Jika tidak ada kecocokan sama sekali
                        if sum(values) == 0:
                            return "netral"
                        # Jika semua nilainya sama
                        if values.count(values[0]) == len(values):
                            return "netral"
                        # Jika tidak, ambil yang jumlahnya paling besar
                        return max(total_match, key=total_match.get)

                    # === Prediksi Gabungan
                    results = []
                    final_predictions = []

                    for i in range(len(cleaned_texts)):
                        cleaned = cleaned_texts[i]
                        tokens = tokenized_texts[i]

                        X_new = vectorizer.transform([cleaned])
                        svm_pred = model.predict(X_new)
                        svm_label = encoder.inverse_transform(svm_pred)[0]

                        lexicon_label = label_by_lexicon(tokens)

                        final_label = lexicon_label if svm_label == 'netral' or svm_label != lexicon_label else svm_label

                        final_predictions.append(svm_label)
                        results.append({
                            "Teks Asli": new_texts[i],
                            "Preprocessed": cleaned,
                            "Prediksi SVM": svm_label,
                            "Prediksi Lexicon": lexicon_label,
                            "Final Decision": final_label
                        })

                    st.subheader("📊 Hasil Prediksi Gabungan (SVM + Lexicon)")
                    st.dataframe(pd.DataFrame(results))

                    y_true = [label_by_lexicon(tokens) for tokens in tokenized_texts]
                    y_pred = final_predictions

                    mapped_labels = ["senang", "marah", "sedih", "netral"]

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

                else:
                    st.error("❌ Gagal membuka halaman URL.")

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")
            finally:
                try:
                    driver.quit()
                except:
                    pass


# ======================================================
# ===================== TAB 2 ==========================
# ======================================================
with tab3:
    st.title("📌 Tab 3: Train Manual dari Dataset Label & Tweet")

    train_file = st.file_uploader("📄 Upload file CSV dengan kolom 'Label' dan 'Tweet'", type=["csv"])

    if train_file:
        try:
            df = pd.read_csv(train_file ,sep=";", encoding="utf-8")

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
