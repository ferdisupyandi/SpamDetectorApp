import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
DATASET_PATH = 'SMSSpamCollection' # Pastika2025n nama file dataset sesuai

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.write("Melatih model deteksi spam... Harap tunggu sebentar.")
    try:
        df = pd.read_csv(DATASET_PATH, sep='\t', header=None, names=['label', 'message'])

        df['label'] = df['label'].map({'ham': 0, 'spam': 1})

        X = df['message']
        y = df['label']

        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_features=5000)
        X_transformed = vectorizer.fit_transform(X)

        model = MultinomialNB()
        model.fit(X_transformed, y)

        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        st.success("Model dan vectorizer berhasil dilatih dan disimpan!")
        
    except FileNotFoundError:
        st.error(f"Error: File dataset '{DATASET_PATH}' tidak ditemukan. Pastikan file berada di direktori yang sama dengan spam.py.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melatih model: {e}")
        st.stop()

st.set_page_config(page_title="SMS Spam Detection", page_icon="📧")

st.markdown("""
<style>
.stTextInput > div > div > input {
    font-size: 1.1rem;
    padding: 0.7rem;
    border-radius: 0.5rem;
}
.stButton > button {
    background-color: #f63366;
    color: white;
    font-size: 1.1rem;
    padding: 0.7rem 2rem;
    border-radius: 0.5rem;
    border: none;
}
.stSuccess {
    background-color: #d4edda;
    color: #155724;
    border-radius: 0.5rem;
    padding: 1rem;
    margin-top: 1rem;
}
.stWarning {
    background-color: #fff3cd;
    color: #856404;
    border-radius: 0.5rem;
    padding: 1rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


st.title("📧 SMS Spam Detection")
st.write("Cek apakah SMS termasuk spam:")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError:
    st.error("File model atau vectorizer tidak ditemukan. Silakan jalankan script secara lokal terlebih dahulu untuk melatih model.")
    st.stop()

user_input = st.text_area("Masukkan pesan SMS:", "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's")

if st.button("Deteksi"):
    if user_input.strip() == "":
        st.warning("Pesan SMS tidak boleh kosong.")
    else:
        input_transformed = vectorizer.transform([user_input])

        prediction = model.predict(input_transformed)
        prediction_proba = model.predict_proba(input_transformed)

        result_label = "Spam" if prediction[0] == 1 else "Bukan Spam"
        result_emoji = "🚫" if prediction[0] == 1 else "✅"
        spam_probability = prediction_proba[0][1]

        st.success(f"Hasil Deteksi: {result_emoji} **{result_label}**")
        st.write(f"Probabilitas Spam: {spam_probability:.2f}")

        st.subheader("Probabilitas Klasifikasi")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Bukan Spam", value=f"{prediction_proba[0][0]*100:.2f}%")
        with col2:
            st.metric(label="Spam", value=f"{prediction_proba[0][1]*100:.2f}%")

        prob_df = pd.DataFrame({
            'Kategori': ['Bukan Spam', 'Spam'],
            'Probabilitas': [prediction_proba[0][0], prediction_proba[0][1]]
        })
        st.bar_chart(prob_df.set_index('Kategori'))