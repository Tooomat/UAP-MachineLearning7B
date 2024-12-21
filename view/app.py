import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from pathlib import Path
import numpy as np
from PIL import Image
import io
import base64

st.set_page_config(page_title="Cornvision", page_icon="🌽")
st.title("CORNVISION")

# CSS 
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: local;
        }}
        .overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 1;
        }}
        .content {{
            position: relative;
            z-index: 2;
            padding: 0px;
            margin: 0px;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: rgba(0, 0, 0, 0.5) !important;
            border: none !important;
            padding: 15px !important;
            width: 100% !important;
            border-radius: 10px !important;
            margin-bottom: 5px !important;
            backdrop-filter: blur(5px) !important;
        }}
        [class="st-emotion-cache-10trblm e1nzilvr1"] {{
            justify-content: center !important;
            text-align: center !important;
            width: 100% !important;
        }}
        [class="st-emotion-cache-10trblm e1nzilvr1"] {{
            justify-content: center !important;
            text-align: center !important;
            width: 100% !important;
        }}
        /* Column container */
        [data-testid="stHorizontalBlock"] {{
            display: flex !important;
            flex-direction: row !important;
            gap: 1rem !important;
            height: auto !important;
        }}
        /* Equal height columns */
        [data-testid="column"] {{
            display: flex !important;
            flex-direction: column !important;
            height: 100% !important;
            background-color: rgba(0, 0, 0, 0.5) !important;
            border-radius: 10px !important;
            backdrop-filter: blur(5px) !important;
            overflow-y: auto !important;
            flex-grow: 1 !important; /* Ensure columns take equal height */
        }}
        /* Ensure content inside columns fills height */
        [data-testid="column"] > div {{
            flex-grow: 1 !important;
            display: flex !important;
            flex-direction: column !important;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# Set background
set_background('static/images/about_bg.jpg')

# Fungsi untuk preprocessing gambar
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224), Image.LANCZOS)
    img_array = keras_image.img_to_array(img)
    img_array = img_array / 255.0  # Normalisasi jika model kamu memerlukan ini
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Fungsi untuk prediksi gambar
def predict_image(img_array, model_path):
    class_names = [
        "Blight", 
        "Common_Rust", 
        "Gray_Leaf_Spot", 
        "Healthy"
    ]

    try:
        model = tf.keras.models.load_model(model_path)
        output = model.predict(img_array)
        score = tf.nn.softmax(output[0])
        print(score)
        confidence = np.max(score)

        if confidence < 0.40:
            return None, confidence

        predicted_class = class_names[np.argmax(score)]
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Terjadi kesalahan saat pemrosesan gambar/prediksi: {e}")
        return None, None


# Fungsi untuk prediksi gambar menggunakan TFLite
def predict_image_tflite(img_array, model_path):
    class_names = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

    try:
        # Load TFLite model and allocate tensors
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        score = tf.nn.softmax(output_data[0])
        print(score)
        confidence = np.max(score)

        if confidence < 0.40:
            return None, confidence

        predicted_class = class_names[np.argmax(score)]
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Terjadi kesalahan saat pemrosesan gambar/prediksi: {e}")
        return None, None

def resize_image(image_path, width, height):
    img = Image.open(image_path)
    img = img.resize((width, height))
    return img

# NAVIGATION
corn_tab, about_tab, deaseas_tab = st.tabs(["Corn", "About", "Deaseas"])

with deaseas_tab:
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    width = 300  # Lebar yang diinginkan
    height = 200 # Tinggi yang diinginkan

    with col1:
        img = resize_image("static/images/Corn_Blight (2).jpg", width, height)
        st.image(img, caption="Corn Blight")
        st.markdown("""
        **Penjelasan:**  
        Corn Blight adalah penyakit yang disebabkan oleh jamur *Bipolaris maydis*. Penyakit ini menyerang daun jagung dan dapat mengurangi hasil panen.  

        **Ciri-ciri:**  
        - Muncul bercak berbentuk elips atau persegi panjang pada daun dengan warna cokelat atau kehitaman.  
        - Bercak sering kali dikelilingi oleh garis kuning (halo).  
        - Infeksi parah dapat menyebabkan daun mengering dan layu.
        """)

    with col2:
        img = resize_image("static/images/Corn_Common_Rust (2).jpg", width, height)
        st.image(img, caption="Corn Common Rust")
        st.markdown("""
        **Penjelasan:**  
        Corn Common Rust disebabkan oleh jamur *Puccinia sorghi*. Penyakit ini umum ditemukan di area dengan iklim lembap dan hangat.  

        **Ciri-ciri:**  
        - Muncul pustula kecil berwarna cokelat kemerahan pada permukaan atas dan bawah daun.  
        - Pustula ini mengandung spora jamur yang mudah tersebar oleh angin.  
        - Jika infeksi parah, daun akan kehilangan klorofil sehingga pertumbuhan tanaman terganggu.
        """)

    with col3:
        img = resize_image("static/images/Corn_Gray_Spot (22).jpg", width, height)
        st.image(img, caption="Corn Gray Spot")
        st.markdown("""
        **Penjelasan:**  
        Penyakit ini disebabkan oleh jamur *Cercospora zeae-maydis*. Gray Leaf Spot biasanya menyerang di akhir musim tanam ketika kelembapan tinggi dan suhu hangat.  

        **Ciri-ciri:**  
        - Bercak kecil berbentuk persegi panjang dengan warna cokelat abu-abu di daun.  
        - Bercak sering kali berbatas tegas, dan bercak yang bertambah besar dapat menyatu menjadi daerah mati yang lebih luas.  
        - Infeksi parah menyebabkan tanaman tidak mampu berfotosintesis dengan baik.
        """)

    with col4:
        img = resize_image("static/images/Corn_Health (11).jpg", width, height)
        st.image(img, caption="Corn Healthy")
        st.markdown("""
        **Penjelasan:**  
        Jagung yang sehat tidak menunjukkan gejala penyakit atau gangguan. Tanaman ini memiliki daun hijau yang segar dan produktivitasnya optimal.  

        **Ciri-ciri:**  
        - Daun berwarna hijau merata tanpa bercak atau kerusakan.  
        - Tanaman tumbuh tegak dengan batang yang kokoh.  
        - Produksi tongkol jagung normal tanpa kelainan atau deformasi.
        """)

# Update state berdasarkan tab yang diklik
# Lanjutan dari update state pada tab 'About'
with about_tab:
    st.header("Selamat Datang Petani Muda 🧑‍🌾")
    st.write("""
        Selamat datang di **Cornvision**, aplikasi berbasis teknologi deep learning 
        yang dirancang untuk membantu petani dan pelaku agribisnis dalam mendeteksi 
        penyakit tanaman jagung secara akurat dan cepat.

        Dengan memanfaatkan model deep learning terkini, Cornvision mampu mengidentifikasi 
        beberapa penyakit pada tanaman jagung hanya dengan menggunakan gambar daun. 
        Teknologi ini diharapkan dapat meningkatkan efisiensi dan efektivitas dalam 
        mengelola tanaman jagung, mengurangi potensi kerugian akibat serangan penyakit, 
        serta membantu pengambilan keputusan yang lebih tepat dalam pengelolaan pertanian.

        **Fitur Utama Cornvision**:
        - Deteksi berbagai penyakit jagung, seperti:
          - **Blight** (Hawar Daun)
          - **Common Rust** (Karat Daun)
          - **Gray Leaf Spot** (Bercak Daun Abu-abu)
          - Daun **Sehat**
        - Mudah digunakan hanya dengan mengunggah citra daun jagung.

        Kami berharap aplikasi ini dapat menjadi solusi inovatif untuk membantu sektor 
        pertanian menjadi lebih modern dan efisien.
    """)

# Update state berdasarkan tab yang diklik
with corn_tab:
    st.header("Ayo Periksakan Tanaman Jagungmu ")
    option = st.selectbox(
        label="Pilih Model",
        options=("Convolutional Neural Network", "MobileNetV2"),
        index=None,
        placeholder="Pilih Metode Yang Akan Digunakan...",
    )

    if option is None:
        st.warning("Pilih Model Terlebih Dahulu!!")
    else:
        col1, col2 = st.columns([3, 2])

        with col1:
            upload = st.file_uploader(
                'Unggah citra untuk mendapatkan hasil prediksi',
                type=['jpg', 'jpeg'])
        with col2:
            st.subheader("Hasil prediksi:")

        if st.button("Predict", type="primary"):
            if upload is not None:
                try:
                    with st.spinner('Memproses citra untuk prediksi..'):
                        image_bytes = upload.getvalue()
                        img_array = preprocess_image(image_bytes)

                        if option == "Convolutional Neural Network":
                            model_path = Path(__file__).parent / "src" / "model_CNN.tflite"
                            result, confidence = predict_image_tflite(img_array, model_path) # Panggil fungsi TFLite
                        elif option == "MobileNetV2":
                            model_path = Path(__file__).parent / "src" / "model_MNV2.hdf5"
                            result, confidence = predict_image(img_array, model_path) # Panggil fungsi Keras
                        else:
                            st.error("Model tidak valid!")
                            st.stop()

                    with col2:
                        img = Image.open(upload)
                        img = img.resize((1000, 500))
                        st.image(img, caption="Gambar yang diunggah")
                        # if result is None:
                        #     st.error("⚠️ Gambar yang diunggah tidak termasuk dalam kategori penyakit yang dikenali!")
                        #     if confidence is not None:
                        #         with col1:
                        #             st.warning(f"""
                        #                 Tingkat keyakinan model: {confidence:.2%}

                        #                 Model saat ini hanya dapat mendeteksi:
                        #                 - Daun jagung yang terinfeksi Blight
                        #                 - Daun jagung yang terinfeksi Common Rust
                        #                 - Daun jagung yang terinfeksi Gray Leaf Spot
                        #                 - Daun jagung sehat
                        #             """)
                        # else:
                        st.success(f"Hasil Prediksi: **{result}**")
                        st.write(f"Tingkat Kepercayaan: **{confidence:.2%}**")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
            else:
                st.warning("Unggah citra terlebih dahulu!!")
