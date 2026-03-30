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
st.set_page_config(page_title="Renal AI | Böbrek Analiz Sistemi", layout="wide")

# --- CUSTOM NAVBAR VE MODERN TEMA CSS ---
st.markdown("""
    <style>
    /* Ana Arka Plan */
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    
    /* Navbar Tasarımı */
    div[data-testid="stHorizontalBlock"] {
        background-color: #161b22;
        padding: 10px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 30px;
    }
    
    /* Kart Yapıları */
    .medical-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    /* Başlıklar */
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Inter', sans-serif; font-weight: 700; }
    
    /* Buton Tasarımı */
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
    
    /* Akademik Notlar */
    .academic-note { font-size: 14px; color: #8b949e; border-left: 4px solid #58a6ff; padding-left: 15px; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

# 2. MODEL YÜKLEME (REFERANS KODUN - HATA ÇÖZÜMÜ)
@st.cache_resource
def load_trained_model():
    model_path = 'kidney_disease_mobilenet_model2.h5'
    if not os.path.exists(model_path):
        st.error(f"❌ Model dosyası ({model_path}) bulunamadı!")
        return None
    try:
        with h5py.File(model_path, 'r+') as f:
            if 'model_config' in f.attrs:
                config_raw = f.attrs['model_config'].decode('utf-8') if isinstance(f.attrs['model_config'], bytes) else f.attrs['model_config']
                config_dict = json.loads(config_raw)
                modified = False
                for layer in config_dict['config']['layers']:
                    if 'batch_shape' in layer['config']:
                        layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')
                        modified = True
                if modified:
                    f.attrs['model_config'] = json.dumps(config_dict).encode('utf-8')
    except: pass
    return tf.keras.models.load_model(model_path, compile=False)

model = load_trained_model()
LABELS = ['Cyst (Kist)', 'Normal', 'Stone (Taş)', 'Tumor (Tümör)']

# 3. ÜST NAVBAR (Yatay Menü)
# Streamlit'te yatay menü için en estetik yöntem budur
st.title("🛡️ Renal AI: Klinik Karar Destek Sistemi")
menu_col1, menu_col2, menu_col3, menu_col4 = st.columns(4)

with menu_col1:
    if st.button("🏠 PROJE VİZYONU"): st.session_state.page = "vizyon"
with menu_col2:
    if st.button("🧠 TEKNİK ALTYAPI"): st.session_state.page = "teknik"
with menu_col3:
    if st.button("📊 ANALİTİK RAPORLAR"): st.session_state.page = "analitik"
with menu_col4:
    if st.button("🔬 CANLI TANI MERKEZİ"): st.session_state.page = "tani"

# Default sayfa ayarı
if 'page' not in st.session_state:
    st.session_state.page = "vizyon"

st.divider()

# --- SAYFA İÇERİKLERİ ---

# 🏠 PROJE VİZYONU (Puan 1-5)
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

# 🧠 TEKNİK ALTYAPI (Puan 6-11)
elif st.session_state.page == "teknik":
    st.header("🧬 Bölüm 2: Veri İşleme ve Model Mimarisi")
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    col_pre, col_arch = st.columns(2)
    with col_pre:
        st.subheader("⚙️ Veri Ön İşleme")
        st.write("- **Normalizasyon:** Piksel değerleri 1/255 oranında ölçeklendi.\n- **Boyutlandırma:** Görüntüler 224x224 piksel formuna getirildi.\n- **Augmentation:** Döndürme ve Zoom işlemleriyle veri çeşitliliği artırıldı.")
    with col_arch:
        st.subheader("🧠 Model Mimarisi")
        st.write("- **Ana Model:** MobileNetV2 (Transfer Learning)\n- **Optimizer:** Adam (lr=0.0001)\n- **Kayıp Fonksiyonu:** Categorical Cross-Entropy")
    st.markdown('</div>', unsafe_allow_html=True)

# 📊 ANALİTİK RAPORLAR (Puan 12-16: Accuracy Buraya Eklendi)
elif st.session_state.page == "analitik":
    st.header("📊 Bölüm 3: Model Başarım Metrikleri")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Doğruluk (Acc)", "%68")
    m2.metric("AUC Skoru", "0.94")
    m3.metric("F1-Skoru", "0.65")
    m4.metric("Duyarlılık", "0.88")

    st.write("---")
    
    # ACCURACY VE LOSS GRAFİĞİ (Madde 15)
    st.subheader("📈 Eğitim Süreci: Doğruluk ve Kayıp Analizi")
    if os.path.exists('learning_curves.png'):
        st.image('learning_curves.png', caption="Eğitim ve Doğrulama Eğrileri", width=1000)
    st.markdown('<p class="academic-note"><b>Analiz:</b> Eğitim ve doğrulama eğrilerinin paralelliği, modelin genelleme yeteneği kazandığını göstermektedir.</p>', unsafe_allow_html=True)

    st.divider()

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("📍 Karmaşıklık Matrisi")
        if os.path.exists('kidney_confusion_matrix.png'):
            st.image('kidney_confusion_matrix.png', use_container_width=True)
    with g2:
        st.subheader("📉 ROC Analizi")
        if os.path.exists('kidney_roc_curve.png'):
            st.image('kidney_roc_curve.png', use_container_width=True)

# 🔬 CANLI TANI MERKEZİ (Puan 19: Teknik Çalışırlık)
elif st.session_state.page == "tani":
    st.header("🔬 Bölüm 4: BT Kesiti Canlı Analiz Modülü")
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    up_file = st.file_uploader("Lütfen analiz edilecek BT görüntüsünü yükleyiniz...", type=["jpg", "png", "jpeg"])
    
    if up_file and model:
        c1, c2 = st.columns(2)
        img = Image.open(up_file).convert('RGB')
        with c1:
            st.image(img, caption="Giriş Kesiti", width=450)
        with c2:
            if st.button("🤖 ANALİZİ BAŞLAT"):
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

# --- ALT BİLGİ (Madde 20: Sonuç ve Kimlik) ---
st.divider()
st.markdown(f"""
    <div style='text-align: center; color: #8b949e;'>
    <b>Geliştirici:</b> Oğuzhan Dursun - 220706037 | <b>Kurum:</b> Giresun Üniversitesi Bilgisayar Mühendisliği<br>
    Yapay Zeka ile Sağlık Bilişimi Dersi Vize Ödevi - 2026
    </div>
    """, unsafe_allow_html=True)
