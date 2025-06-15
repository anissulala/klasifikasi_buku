import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
from tfrf import TFRFVectorizerMulticlass

# Konfigurasi halaman
st.set_page_config(page_title="Web Buku", layout="centered")

# Tambahkan CSS gabungan (final)
st.markdown("""
    <style>
        .block-container { 
            background-color: #ffffff !important; 
            padding-top: 1rem !important; 
            padding-bottom: 1rem !important; 
        }
        .top-nav { 
            background-color: #6C2BD9; 
            display: flex; 
            justify-content: flex-start; 
            align-items: center; 
            height: 60px; 
            padding: 0 40px; 
            gap: 32px; 
            font-weight: bold; 
            font-size: 18px; 
            font-family: sans-serif; 
            margin-bottom: 40px; 
        }
        .top-nav a { 
            color: white; 
            text-decoration: none; 
        }
        .top-nav a.selected { 
            text-decoration: underline; 
        }
        .form-container { 
            margin-top: 10px; 
            font-family: sans-serif; 
            text-align: left; 
            margin-left: 30%; 
            margin-right: 30%; 
        }
        div[data-testid="stForm"] { 
            background-color: transparent !important; 
            border: none !important; 
            box-shadow: none !important; 
            padding: 0 !important; 
        }
        div[data-testid="stFormContainer"] { 
            padding: 0 !important; 
        }
        .main-title { 
            text-align: center; 
            color: #333; 
            font-size: 28px; 
            margin-bottom: 20px; 
        }
        .input-label { 
            margin-bottom: 5px; 
            font-size: 16px; 
        }
        .input-row input { 
            border-radius: 10px; 
            padding: 10px 14px; 
            width: 100%; 
            border: 1px solid #ccc; 
            font-size: 16px; 
        }
        .stButton>button { 
            margin-top: 8px; 
            padding: 15px 30px; 
            background-color: #6C2BD9; 
            border: none; 
            border-radius: 12px; 
            cursor: pointer; 
            font-weight: bold; 
            color: white; 
            font-size: 16px; 
            width: 100%;
        }
        .result-container { 
            margin-top: 20px; 
            padding: 20px; 
            background-color: #f8f9fa; 
            border-radius: 12px; 
            border: 1px solid #e0e0e0; 
        }
    </style>
""", unsafe_allow_html=True)

# Load model, vectorizer, dan label encoder
database = pd.read_csv('percobaan_processed.csv')
database.dropna(inplace=True)
jumlah = database['label'].value_counts().reset_index().rename(columns={'count':'jumlah'})
map_label = {int(jumlah.label[i]): int(jumlah.jumlah[i]) for i in range(len(jumlah))}
database['value'] = database.label.map(map_label)
database = database[database.value > 20]
database = database[database.label != 40]
data_text = database['stemmed_title']
kelas_judul = database['label']

tf_rf = joblib.load('tf_rf.pkl')
label = joblib.load('le.pkl')
model = joblib.load('knn_fix.pkl')

stopword = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(text):
    return " ".join([stemmer.stem(word) for word in text])

def preprocess_judul(text):
    text = text.lower().split()
    text = [word for word in text if word not in stopword]
    return stemming(text)

ddc_data = pd.read_excel('keterangan DDC.xlsx', sheet_name='data 3')
ddc_data.dropna(inplace=True)
#
dict_ddc = {int(ddc_data['Klasifikasi'][i]): ddc_data['DDC '][i] for i in range(len(ddc_data))}

# Inisialisasi session state
if "riwayat_klasifikasi" not in st.session_state:
    st.session_state.riwayat_klasifikasi = []

# Sidebar Menu
menu = st.sidebar.radio("Menu", ["Beranda", "Klasifikasi"])

# Halaman BERANDA
if menu == "Beranda":
    st.markdown('<h1 class="main-title">Klasifikasi Judul Buku Perpustakaan</h1>', unsafe_allow_html=True)
    st.markdown('<div class="form-container">', unsafe_allow_html=True)

    with st.form(key='book_form'):
        st.markdown('<div class="input-label">Input Judul Buku</div>', unsafe_allow_html=True)
        judul = st.text_input("", placeholder="Masukkan judul buku", label_visibility="collapsed")
        submitted = st.form_submit_button("Kirim")

        if submitted and judul.strip():
            try:
                input_judul = preprocess_judul(judul)
                # tf_rf.fit(data_text, y=kelas_judul)
                tf_rf_judul = tf_rf.transform([input_judul])
                predict = model.predict(tf_rf_judul)
                fix_predict = label.inverse_transform(predict)[0]


                # Simpan hasil klasifikasi ke session state
                st.session_state.riwayat_klasifikasi.append({
                    "judul": judul,
                    "hasil": fix_predict
                })
                
                st.success(f"✅ Buku '{judul}' berhasil diklasifikasikan sebagai kategori '{fix_predict} yaitu {dict_ddc[fix_predict]}'")

            except Exception as e:
                st.error(f"Error saat klasifikasi: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# Halaman KLASIFIKASI
elif menu == "Klasifikasi":
    st.markdown('<h1 class="main-title">Hasil Klasifikasi Buku</h1>', unsafe_allow_html=True)

    if st.session_state.riwayat_klasifikasi:
        
        # Tampilkan jumlah buku
        jumlah_buku = len(st.session_state.riwayat_klasifikasi)
        st.markdown(f"""
            <div style="font-size: 18px; font-weight: bold; margin-bottom: 20px;">
                Jumlah Buku: {jumlah_buku}
            </div>
        """, unsafe_allow_html=True)

        
        for idx, item in enumerate((st.session_state.riwayat_klasifikasi), 1):
            st.markdown(f"""
                <div class="result-container" style="margin-bottom: 15px;">
                    <div style="font-size: 16px; margin-bottom: 8px;">
                        <strong>#{idx}. {item['judul']}</strong>
                    </div>
                    <div style="font-size: 14px; color: #6C2BD9; font-weight: bold;">
                        Kategori: {item['hasil']} yaitu {dict_ddc[item['hasil']]}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Hapus Semua Riwayat"):
            st.session_state.riwayat_klasifikasi = []
            st.rerun()
    else:
        st.info("Belum ada hasil klasifikasi. Silakan inputkan data di Beranda.")