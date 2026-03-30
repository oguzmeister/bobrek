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

# 1. SAYFA YAPILANDIRMASI (Oğuzhan Dursun - 220706037)
st.set_page_config(page_title="Renal AI | Böbrek Analiz Pro", layout="wide")

# --- GELİŞMİŞ TIBBİ PANEL CSS ---
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
    .stButton>button:hover { transform: scale(1.01); box-shadow: 0 5px 15px rgba(88, 166, 255, 0.4); }
    .academic-note { font-size: 14px; color: #8b949e; border-left: 4px solid #58a6ff; padding-left: 15px; font-style: italic; }
    /* Tahmin Kartı Stili */
    .prediction-box {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 12px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. MODEL YÜKLEME (HATA ÇÖZÜMÜ DAHİL)
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
                    if 'config' in layer and 'batch_shape' in layer['config']:
                        layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')
                        modified = True
                if modified:
                    f.attrs['model_config'] = json.dumps(config_dict).encode('utf-8')
    except: pass

    from tensorflow.keras.layers import InputLayer
    class CompatibleInputLayer(InputLayer):
        def __init__(self, *args, **kwargs):
            if 'batch_shape' in kwargs:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
            super().__init__(*args, **kwargs)

    try:
        return tf.keras.models.load_model(model_path, compile=False, custom_objects={'InputLayer': CompatibleInputLayer})
    except Exception: return None

model = load_eye_model() if 'load_eye_model' in locals() else load_trained_model()
LABELS = ['Cyst (Kist)', 'Normal', 'Stone (Taş)', 'Tumor (Tümör)']

# 3. ÜST NAVBAR (Yatay Menü)
st.title("🛡️ Renal AI: Böbrek Patolojisi Analiz Pro")
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
        st.write("Böbrek patolojilerinin BT kesitlerinden teşhisi, uzman radyologlar için bile zaman alıcı bir süreçtir. Renal AI, klinik karar destek mekanizması olarak teşhis doğruluğunu artırır.")
    with c2:
        st.subheader("📚 Veri Seti")
        st.info("**Kaynak:** Kaggle CT-Kidney\n\n**Sınıf:** 4 (Kist, Normal, Taş, Tümör)")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "teknik":
    st.header("🧬 Bölüm 2: Veri İşleme ve Model Mimarisi")
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    col_pre, col_arch = st.columns(2)
    with col_pre:
        st.subheader("⚙️ Veri Ön İşleme")
        st.write("- **Normalizasyon:** 1/255 ölçeklendirme.\n- **Boyutlandırma:** 224x224 piksel.\n- **Augmentation:** Döndürme ve Zoom.")
    with col_arch:
        st.subheader("🧠 Model Mimarisi")
        st.write("- **Ana Model:** MobileNetV2\n- **Optimizer:** Adam (lr=0.0001)\n- **Kayıp:** Categorical Cross-Entropy")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.page == "analitik":
    st.header("📊 Bölüm 3: Model Başarım Metrikleri")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Doğruluk", "%68")
    m2.metric("AUC Skoru", "0.94")
    m3.metric("F1-Skoru", "0.65")
    m4.metric("Duyarlılık", "0.88")
    st.divider()
    st.subheader("📈 Eğitim Süreci: Doğruluk ve Kayıp Analizi")
    if os.path.exists('learning_curves.png'):
        st.image('learning_curves.png', caption="Eğitim ve Doğrulama Eğrileri", width=1000)
    g1, g2 = st.columns(2)
    with g1:
        if os.path.exists('kidney_confusion_matrix.png'): st.image('kidney_confusion_matrix.png', caption="Karmaşıklık Matrisi", use_container_width=True)
    with g2:
        if os.path.exists('kidney_roc_curve.png'): st.image('kidney_roc_curve.png', caption="ROC Analizi", use_container_width=True)

# 🔬 CANLI TANI MERKEZİ (DÜZENLENEN KISIM)
elif st.session_state.page == "tani":
    st.header("🔬 Bölüm 4: BT Kesiti Canlı Analiz Modülü")
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    up_file = st.file_uploader("Analiz edilecek BT görüntüsünü (JPG/PNG) yükleyiniz...", type=["jpg", "png", "jpeg"])
    
    if up_file and model:
        # Görsel ve Analizi yan yana getiren sütunlar
        col_img, col_analysis = st.columns([1, 1], gap="large")
        
        img = Image.open(up_file).convert('RGB')
        
        with col_img:
            st.markdown("### 🖼️ Giriş BT Kesiti")
            # Container büyümesini önlemek için width sabitleme
            st.image(img, caption="Yüklenen Ham Görüntü", width=450)
        
        with col_analysis:
            st.markdown("### 🤖 Analiz Sonuçları")
            if st.button("ANALİZİ BAŞLAT"):
                with st.spinner('Yapay Zeka Taraması Yapılıyor...'):
                    # İşleme
                    p_img = np.array(img.resize((224, 224))).astype('float32') / 255.0
                    p_img = np.expand_dims(p_img, axis=0)
                    preds = model.predict(p_img, verbose=0)
                    idx = np.argmax(preds)
                    
                    # Tahmin Kartı
                    res_color = "#238636" if idx == 1 else "#da3633"
                    st.markdown(f"""
                        <div class="prediction-box" style="border-left: 10px solid {res_color};">
                            <h2 style="margin:0;">Teşhis: {LABELS[idx]}</h2>
                            <h3 style="margin:0; color: #8b949e !important;">Güven Oranı: %{np.max(preds)*100:.1f}</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Olasılık Dağılım Grafiği
                    st.write("")
                    df_p = pd.DataFrame({'Patoloji': LABELS, 'Olasılık': preds[0]})
                    fig = px.bar(df_p, x='Patoloji', y='Olasılık', color='Patoloji', height=300,
                                 template="plotly_dark", color_discrete_sequence=px.colors.sequential.Blues_r)
                    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- ALT BİLGİ ---
st.divider()
st.markdown(f"<div style='text-align: center; color: #8b949e;'><b>Geliştirici:</b> Oğuzhan Dursun - 220706037 | <b>Ders:</b> Yapay Zeka ile Sağlık Bilişimi Vize Ödevi - 2026</div>", unsafe_allow_html=True)
