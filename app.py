import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import h5py
import json

# 1. SAYFA YAPILANDIRMASI (Madde 17 & 18: Navbar ve Modern Arayüz)
st.set_page_config(page_title="Renal AI | Böbrek Analiz Pro", layout="wide")

# --- CUSTOM NAVBAR VE MODERN TEMA CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    div[data-testid="stHorizontalBlock"] {
        background-color: #161b22;
        padding: 10px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 30px;
    }
    .medical-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Inter', sans-serif; font-weight: 700; }
    .stButton>button {
        background: linear-gradient(90deg, #1d4ed8, #2563eb);
        color: white;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        height: 50px;
        width: 100%;
        transition: 0.4s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 5px 15px rgba(88, 166, 255, 0.4); }
    .academic-note { font-size: 14px; color: #8b949e; border-left: 4px solid #58a6ff; padding-left: 15px; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

# 2. MODEL YÜKLEME (HATAYI ÇÖZEN GÜNCELLENMİŞ MANTIK)
@st.cache_resource
def load_trained_model():
    model_path = 'kidney_disease_mobilenet_model2.h5'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model dosyası ({model_path}) bulunamadı!")
        return None

    # H5 Dosyasındaki Yapılandırma Hatasını Düzelten Yama
    try:
        # Modeli r+ (okuma-yazma) modunda açıp config'i Keras 3'e uyarlıyoruz
        with h5py.File(model_path, 'r+') as f:
            if 'model_config' in f.attrs:
                config_raw = f.attrs['model_config']
                if isinstance(config_raw, bytes):
                    config_raw = config_raw.decode('utf-8')
                
                config_dict = json.loads(config_raw)
                modified = False
                
                # Katmanları tarayıp 'batch_shape'i 'batch_input_shape'e çeviriyoruz
                if 'config' in config_dict and 'layers' in config_dict['config']:
                    for layer in config_dict['config']['layers']:
                        if 'config' in layer and 'batch_shape' in layer['config']:
                            layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')
                            modified = True
                
                if modified:
                    f.attrs['model_config'] = json.dumps(config_dict).encode('utf-8')
    except Exception as e:
        # Dosya salt okunur olabilir (Cloud üzerinde normaldir), hatayı sessizce geçiyoruz
        pass

    # Keras 3 Uyumlu Katman Yaması (Zorunlu)
    from tensorflow.keras.layers import InputLayer
    class CompatibleInputLayer(InputLayer):
        def __init__(self, *args, **kwargs):
            if 'batch_shape' in kwargs:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
            super().__init__(*args, **kwargs)

    try:
        # custom_objects kullanarak Keras'a yeni katman mantığını tanıtıyoruz
        return tf.keras.models.load_model(
            model_path, 
            compile=False, 
            custom_objects={'InputLayer': CompatibleInputLayer}
        )
    except Exception as e:
        st.error(f"❌ Model Yapılandırma Hatası Devam Ediyor: {e}")
        return None

# Model Belleğe Alınır
model = load_trained_model()
LABELS = ['Cyst (Kist)', 'Normal', 'Stone (Taş)', 'Tumor (Tümör)']

# 3. ÜST NAVBAR (Yatay Menü)
st.title("🛡️ Renal AI: İleri Seviye Böbrek Analiz Sistemi")
menu_col1, menu_col2, menu_col3, menu_col4 = st.columns(4)

with menu_col1:
    if st.button("🏠 PROJE VİZYONU"): st.session_state.page = "vizyon"
with menu_col2:
    if st.button("🧠 TEKNİK ALTYAPI"): st.session_state.page = "teknik"
with menu_col3:
    if st.button("📊 ANALİTİK RAPORLAR"): st.session_state.page = "analitik"
with menu_col4:
    if st.button("🔬 CANLI TANI MERKEZİ"): st.session_state.page = "tani"

if 'page' not in st.session_state: st.session_state.page = "vizyon"
st.divider()

# --- SAYFA İÇERİKLERİ ---

if st.session_state.page == "vizyon":
    st.header("🏥 Bölüm 1: Problemin Tanımı ve Akademik Önemi")
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📌 Problemin Tanımı")
        st.write("""
        Böbrek patolojilerinin Bilgisayarlı Tomografi (BT) kesitlerinden teşhisi, radyologlar için yüksek konsantrasyon gerektiren kritik bir süreçtir. 
        Bu proje, derin öğrenme temelli bir yapay zeka sistemi geliştirerek, radyolojik görüntülemede klinik karar destek mekanizması oluşturmayı hedefler.
        Sistem, kist ve tümör gibi yapıları otomatik olarak saptayarak hekimlerin teşhis hızını ve doğruluğunu artırır.
        """)
    with c2:
        st.subheader("📚 Veri Seti")
        st.info("**Kaynak:** Kaggle CT-Kidney\n\n**Kapsam:** 12.000+ Görüntü\n\n**Sınıflar:** Normal, Kist, Taş, Tümör")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "teknik":
    st.header("🧬 Bölüm 2: Veri İşleme ve Model Mimarisi")
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    col_pre, col_arch = st.columns(2)
    with col_pre:
        st.subheader("⚙️ Veri Ön İşleme")
        st.write("- **Normalizasyon:** Piksel değerleri [0, 1] arasına ölçeklendi.\n- **Boyutlandırma:** Görüntüler 224x224 piksel formuna getirildi.\n- **Augmentation:** Döndürme ve Zoom işlemleriyle veri çeşitliliği artırıldı.")
    with col_arch:
        st.subheader("🧠 Model Mimarisi")
        st.write("- **Ana Model:** MobileNetV2 (Transfer Learning)\n- **Optimizer:** Adam (lr=0.0001)\n- **Kayıp Fonksiyonu:** Categorical Cross-Entropy")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "analitik":
    st.header("📊 Bölüm 3: Model Başarım Metrikleri")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Genel Doğruluk", "%68")
    m2.metric("AUC Skoru", "0.94")
    m3.metric("F1-Skoru", "0.65")
    m4.metric("Duyarlılık", "0.88")
    
    st.divider()
    
    st.subheader("📈 Eğitim Süreci Analizi (Doğruluk ve Kayıp)")
    if os.path.exists('learning_curves.png'):
        st.image('learning_curves.png', caption="Eğitim ve Doğrulama Eğrileri", width=1000)
    st.markdown('<p class="academic-note"><b>Analiz:</b> Doğruluk ve kayıp eğrilerinin paralelliği, modelin genelleme yeteneğini ve öğrenme stabilitesini kanıtlamaktadır.</p>', unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("📍 Karmaşıklık Matrisi")
        if os.path.exists('kidney_confusion_matrix.png'):
            st.image('kidney_confusion_matrix.png', use_container_width=True)
    with g2:
        st.subheader("📉 ROC Analizi")
        if os.path.exists('kidney_roc_curve.png'):
            st.image('kidney_roc_curve.png', use_container_width=True)

elif st.session_state.page == "tani":
    st.header("🔬 Bölüm 4: BT Kesiti Canlı Analiz Modülü")
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    up_file = st.file_uploader("Lütfen analiz edilecek BT görüntüsünü yükleyiniz...", type=["jpg", "png", "jpeg"])
    
    if up_file and model:
        c1, c2 = st.columns(2)
        img = Image.open(up_file).convert('RGB')
        with c1:
            st.image(img, caption="Giriş BT Kesiti", width=450)
        with c2:
            if st.button("🤖 ANALİZİ BAŞLAT"):
                with st.spinner('Yapay Zeka Taraması Yapılıyor...'):
                    p_img = np.array(img.resize((224, 224))).astype('float32') / 255.0
                    p_img = np.expand_dims(p_img, axis=0)
                    preds = model.predict(p_img, verbose=0)
                    idx = np.argmax(preds)
                    
                    st.subheader("Tahhis Sonucu")
                    res_color = "#238636" if idx == 1 else "#da3633"
                    st.markdown(f"<h1 style='color: {res_color};'>{LABELS[idx]}</h1>", unsafe_allow_html=True)
                    st.metric("Güven Oranı", f"%{np.max(preds)*100:.1f}")
                    
                    st.plotly_chart(px.bar(pd.DataFrame({'Patoloji': LABELS, 'Olasılık': preds[0]}), 
                                           x='Patoloji', y='Olasılık', color='Patoloji', template="plotly_dark"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 4. ALT BİLGİ (Madde 20: Akademik Kimlik)
st.divider()
st.markdown(f"""
    <div style='text-align: center; color: #8b949e;'>
    <b>Geliştirici:</b> Oğuzhan Dursun - 220706037 | <b>Bölüm:</b> Bilgisayar Mühendisliği<br>
    <b>Ders:</b> Yapay Zeka ile Sağlık Bilişimi Vize Ödevi - 2026
    </div>
    """, unsafe_allow_html=True)
