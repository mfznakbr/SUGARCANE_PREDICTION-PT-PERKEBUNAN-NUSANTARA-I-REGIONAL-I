import pandas as pd 
import numpy as np
import joblib
from joblib import load

# DATA TEST FOR MODEL
df = pd.read_excel(r'D:\portofolio\SUGARCANE_PREDICTION\FILE\data_testing.xlsx')
data = df.sample(n=5, random_state=42)
print(data)

# PREPROCESSING
def processing(data):
    load_path = "D:\PROJECT PORTOFOLIO\SUGARCANE_PREDICTION\FILE\preproses_data.joblib"
    prepro = load(load_path)
    # print(f"pipeline preprocessing dimuat dari : {load_path}")

    transformed_data = prepro.transform(data)
    # print("Data setelah preprocessing:")
    # print(transformed_data[:5])  # Print hanya 5 baris pertama
    return transformed_data

# PREDICTION
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        # print("✅ Model berhasil dimuat!")
        return model
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
        return None
    
# Main for prediction
model_path = r"D:\PROJECT PORTOFOLIO\SUGARCANE_PREDICTION\FUNCTION\mlartifacts\701912226840097472\models\m-38ffa3d426c845d186844fc92f37caf0\artifacts\model.pkl"
def main(data):
    # load model
    model = load_model(model_path)
    sample_data = processing(data)  # Preprocessing sesuai pipeline
    if model is None:
        return
    prediksi = model.predict(sample_data)
    return prediksi

if __name__ == "__main__":
    result = main(data)
    print(result)