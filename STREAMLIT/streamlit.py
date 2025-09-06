import streamlit as st
import pandas as pd
import numpy as np

from joblib import load
import json
from test_model import main, processing, load_model

# LOGO AND TITLE
st.set_page_config(page_title="Sugarcane Yield Prediction", page_icon="🌾", layout="wide")
st.title("🌾 Sugarcane Yield Prediction App")
# st.image("D:/PROJECT PORTOFOLIO/SUGARCANE_PREDICTION/ASET_DEPLOY/cover_tebu.png")

col1, col2 = st.columns([3, 2])

with col1:

    # Input for categorical data
    DP = st.selectbox("DP", ["I", "II", "III", "IV", "V"])
    Rayon = st.selectbox("Rayon", ["A", "B"])
    Tingkat_tanam = st.selectbox("Tingkat Tanam", ["PC", "R1", "R2", "R3"])
    bulan_tanam = st.selectbox("Bulan Tanam", ["1A", "1B", "2A", "2B", "3A", "3B", "4A", "4B", "5A", "5B", "6A", "6B", "7A", "7B"])
    variaetas = st.selectbox("Varietas", ["BZ 134", "PS 094", "PSJT 941", "KK"])
    Kategori = st.selectbox("Kategpri", ["Manual", "Mekanisasi"])

    # Input for numerical data
    Faktor_juring = st.number_input("Faktor_juring", min_value=0, value=7400)
    luas = st.number_input("Luas (Ha) ", min_value=0.0, value=4.0)
    tahun = st.number_input("Tahun ", min_value=0, value=2023)
    jlh_batang  = st.number_input("Jumlah Batang per M Juring", min_value=0.0, value=4.02)
    Bobot = st.number_input("Bobot per meter Batang (Kg)", min_value=0.0, value=0.41)
    panjang_ini = st.number_input("Rata-Rata Panjang Saat Ini (m)", min_value=0.0, value=1.12)
    panjang_akhir = st.number_input("Rata-Rata Panjang Akhir (m)", min_value=0.0, value=2.62)  

    # Combine inputs into a dictionary (single row of data)
    input_dict = {
    "DP": DP,
    "Tingkat Tanam": Tingkat_tanam,
    "Rayon": Rayon,
    "Tahun" : int(tahun),
    "Kategori": Kategori,
    "Luas (Ha)": float(luas),
    "Bulan Tanam": bulan_tanam,
    "Varietas": variaetas,
    "Faktor Juring (m)": int(Faktor_juring),
    "Jumlah Batang per M Juring": float(jlh_batang),
    "Bobot per meter Batang (Kg)": float(Bobot),
    "Rata-Rata Panjang Saat Ini (m)": float(panjang_ini),
    "Rata-Rata Panjang Akhir (m)": float(panjang_akhir),     # dummy value untuk placeholder target (bisa di-drop nanti)
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_dict])

    # Button to display data
    if st.button("Simpan & Tampilkan Data"):
        st.write("### Data yang Dimasukkan:")
        st.dataframe(df)

        # Preprocessing
        new_data = processing(data=df)  # This returns an ndarray
        st.write("### Data setelah diolah:")
        st.write(pd.DataFrame(new_data))

        # Prediction
        result = main(df)
        st.write("### Hasil Prediksi (Ton/Ha):")
        st.success(f"Predicted Sugarcane Yield: {result[0]:.2f} Ton/Ha")


# DESCRIPTION OF APP
with col2:
    st.write("## About this App")
    # st.image("D:\PROJECT PORTOFOLIO\SUGARCANE_PREDICTION\ASET_DEPLOY\mlTebu.png")
    st.write("""
    This application predicts the sugarcane yield based on various agricultural and environmental factors. 
    It utilizes a machine learning model trained on historical data to provide accurate yield predictions.
    
    **How it works:**
    1. The model uses historical plantation data.
    2. Input features such as *DP, Rayon, Planting Level, and Planting Month* are encoded and processed.
    3. Numerical features are scaled to ensure optimal model performance.
    4. The processed data is fed into a pre-trained Random Forest model to generate yield predictions.
    5. The predicted yield is displayed in tons (Ton).
             
    **Why this matters:**
    - This application addresses a real issue at **PTPN I Regional 1**, where yield predictions were traditionally conducted manually.  
    - Such manual methods frequently led to **significant errors** that affected planning and operational efficiency.  
    - By employing a **data-driven machine learning approach**, this app aims to deliver **more accurate and reliable predictions**.  
    - Ultimately, it supports **strategic planning, resource optimization, and sustainable sugarcane production management**.
    
    **Note:** Ensure that all inputs are accurate to get the best prediction results.
    """)

# MODEL EVALUATION
st.subheader("Model Evaluation")
