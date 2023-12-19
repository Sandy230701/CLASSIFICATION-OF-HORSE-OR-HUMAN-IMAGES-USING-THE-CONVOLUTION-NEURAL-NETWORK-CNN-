import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load the pre-trained model
model = load_model('model.h5')  # Ganti 'model.h5' dengan nama model yang sesuai

st.title("Deteksi Gambar")

uploaded_file = st.file_uploader("Pilih gambar...")

if uploaded_file is not None:
    # Membaca dan menampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)

    # Praproses gambar untuk membuat prediksi
    img = load_img(uploaded_file, target_size=(150, 150))
    x = img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)

    # Membuat prediksi menggunakan model
    prediction = model.predict(x)

    # Menampilkan hasil prediksi dan akurasi
    st.subheader("Hasil Prediksi:")
    if prediction[0] > 0.5:
        st.write("Ini adalah manusia.")
    else:
        st.write("Ini adalah kuda.")

    # Menampilkan akurasi prediksi
    accuracy = prediction[0] if prediction[0] > 0.5 else 1 - prediction[0]
    st.write("Akurasi prediksi: ", accuracy )
