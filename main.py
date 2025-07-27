import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import re
import time
from urllib.parse import urlparse
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

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
tab1, tab2 = st.tabs(["📌 Scraping & Proses Lexicon + TF-IDF + SVM", "📌 Train Manual dari Label & Tweet"])

with tab1:
    st.title("📌Proses Otomatis dari Scraping Tokopedia (Lexicon + TF-IDF + SVM)")

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
                        with open(f'model/tfidf_vectorizer_{kernel_option}.pkl', 'rb') as f:
                            vectorizer = pickle.load(f)
                        with open(f'model/label_encoder_{kernel_option}.pkl', 'rb') as f:
                            encoder = pickle.load(f)
                    except FileNotFoundError:
                        st.error(f"❌ Model untuk kernel '{kernel_option}' belum dilatih. Silakan latih terlebih dahulu di Tab 3 atau pastikan file 'svm_model_{kernel_option}.pkl' ada.")
                        st.stop()

                    # === Load Lexicon (senang, marah, sedih saja)
                    nrc_df = pd.read_csv("lexicon/Indonesian-NRC-EmoLex.csv", sep=";", encoding="utf-8")
                    senang_words = set(nrc_df[(nrc_df['joy'] == 1) | (nrc_df['positive'] == 1)]['Indonesian Word'].str.lower())
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

                    emotion_counts = Counter(y_pred)
                    df_emotion = pd.DataFrame(emotion_counts.items(), columns=['Emosi', 'Jumlah'])

                    # Hitung total jumlah
                    st.subheader("📉 Diagram Batang Emosi Prediksi")
                    fig2, ax2 = plt.subplots()
                    # Hitung total jumlah
                    total = df_emotion['Jumlah'].sum()
                    # Buat bar plot
                    sns.barplot(data=df_emotion, x='Emosi', y='Jumlah', palette="pastel", ax=ax2)
                    # Tambahkan teks jumlah dan persentase di atas setiap bar
                    for i, row in df_emotion.iterrows():
                        # Hitung persentase
                        percentage = (row['Jumlah'] / total) * 100
                        
                        # Tampilkan nilai jumlah dan persentase
                        ax2.text(i, 
                                row['Jumlah'] + 0.5, 
                                f"{int(row['Jumlah'])}\n({percentage:.1f}%)",  # Gabungkan nilai dan persentase
                                ha='center', 
                                va='center',
                                fontsize=10)
                    # Atur label sumbu dan judul jika diperlukan
                    ax2.set_ylabel('Jumlah')
                    ax2.set_xlabel('Emosi')
                    ax2.set_title('Distribusi Emosi')
                    st.pyplot(fig2)
               

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
with tab2:
    st.title("📚 Train SVM Emosi (Multi Kernel)")
    uploaded_file = st.file_uploader("📁 Upload dataset CSV", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file, sep=';')
        
        if 'text' not in data.columns or 'emotion' not in data.columns:
            st.error("❌ CSV harus punya kolom 'text' dan 'emotion'")
        else:
            st.success("✅ Dataset berhasil dimuat!")
            st.dataframe(data)

            if st.button("🚀 Mulai Training"):
                with st.spinner("🔄 Memproses data..."):
                    texts = data['text']
                    labels = data['emotion']

                    cleaned_texts = texts.apply(preprocess)
                    vectorizer = TfidfVectorizer()
                    X = vectorizer.fit_transform(cleaned_texts)

                    encoder = LabelEncoder()
                    y = encoder.fit_transform(labels)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.5, random_state=42, stratify=y
                    )

                    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
                    os.makedirs('model', exist_ok=True)

                    for kernel in kernels:
                        st.subheader(f"🔧 Kernel: {kernel}")
                        model = SVC(kernel=kernel, probability=True)
                        model.fit(X_train, y_train)
                        # Simpan model
                        with open(f'model/svm_model_{kernel}.pkl', 'wb') as f:
                            pickle.dump(model, f)
                        with open(f'model/tfidf_vectorizer_{kernel}.pkl', 'wb') as f:
                            pickle.dump(vectorizer, f)
                        with open(f'model/label_encoder_{kernel}.pkl', 'wb') as f:
                            pickle.dump(encoder, f)

                        st.success(f"✅ Model kernel '{kernel}' berhasil disimpan!")