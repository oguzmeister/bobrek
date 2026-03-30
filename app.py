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

# 1. SAYFA YAPILANDIRMASI (Madde 17 & 18: Navigasyon ve Estetik)
st.set_page_config(page_title="Renal AI | Böbrek Analiz Pro", layout="wide")

# --- GELİŞMİŞ TIBBİ PANEL CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .main-card { background: #1f2937; border: 1px solid #38444d; padding: 25px; border-radius: 12px; margin-bottom: 20px; }
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Inter', sans-serif; font-weight: 700; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    .stButton>button { background: linear-gradient(90deg, #1d4ed8, #2563eb); color: white; border-radius: 8px; font-weight: bold; border: none; height: 50px; width: 100%; transition: 0.3s; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(29, 78, 216, 0.4); }
    .academic-label { font-size: 14px; color: #8b949e; border-left: 3px solid #58a6ff; padding-left: 10px; margin: 10px 0; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

# 2. MODEL YÜKLEME (SENİN REFERANS KODUN)
@st.cache_resource
def load_trained_model():
    model_path = 'kidney_disease_mobilenet_model2.h5'
    if not os.path.exists(model_path):
        st.error(f"❌ Model dosyası ({model_path}) bulunamadı!")
        return None
    try:
        with h5py.File(model_path, 'r+') as f:
            if 'model_config' in f.attrs:
                config_raw = f.attrs['model_config']
                if isinstance(config_raw, bytes):
                    config_raw = config_raw.decode('utf-8')
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

try:
    model = load_trained_model()
    LABELS = ['Cyst (Kist)', 'Normal', 'Stone (Taş)', 'Tumor (Tümör)']
except Exception as e:
    st.error(f"⚠️ Model Dosyası Hatası: {e}")

# 3. NAVİGASYON PANELİ (SIDEBAR)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2864/2864283.png", width=100)
    st.title("Renal AI")
    st.markdown("---")
    # Navbar Seçenekleri
    page = st.radio("Sistem Menüsü", ["🏠 Proje Özeti", "🧬 Teknik Metodoloji", "📊 Performans Analizi", "🔬 Canlı Tanı Laboratuvarı"])
    st.markdown("---")
    st.info(f"**Geliştirici:** Oğuzhan Dursun\n\n**Öğrenci No:** 220706037\n\n**Bölüm:** Bilgisayar Mühendisliği")
    st.caption("Yapay Zeka ile Sağlık Bilişimi Dersi Vize Ödevi - © 2026")

# --- SAYFA İÇERİKLERİ ---

# 🏠 PROJE ÖZETİ
if page == "🏠 Proje Özeti":
    st.header("🔬 Bölüm 1: Klinik Tanımlama ve Hedefler")
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("📌 Problemin Akademik Tanımı")
        st.write("""
        Böbrek patolojilerinin Bilgisayarlı Tomografi (BT) kesitlerinden teşhisi, uzman radyologlar için bile yüksek dikkat ve zaman gerektiren bir süreçtir. 
        Sağlık Bilişimi disiplini altında geliştirilen bu proje, derin öğrenme algoritmalarını kullanarak BT görüntülerinde kist, taş ve tümör varlığını 
        otomatik olarak analiz eder. Sistem, radyologlar için bir 'İkinci Göz' (Second Opinion) mekanizması olarak çalışarak tanı hatalarını minimize etmeyi amaçlar.
        """)
    with col_b:
        st.subheader("📚 Veri Kaynağı")
        st.success("**Dataset:** Kaggle CT-Kidney\n\n**Kapsam:** 12.000+ DICOM Görüntüsü\n\n**Sınıf Sayısı:** 4 (Normal, Kist, Taş, Tümör)")
    st.markdown('</div>', unsafe_allow_html=True)

# 🧬 TEKNİK METODOLOJİ
elif page == "🧬 Teknik Metodoloji":
    st.header("🧬 Bölüm 2: Mühendislik Yaklaşımı ve Mimari")
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    t1, t2 = st.tabs(["⚙️ Veri Ön İşleme", "🧠 Model Parametreleri"])
    with t1:
        st.write("""
        1. **Geometrik Standardizasyon:** Görüntüler 224x224 piksel boyutuna normalize edildi.
        2. **Piksel Ölçeklendirme:** [0, 1] aralığında normalizasyon (1/255) uygulandı.
        3. **Veri Artırımı (Augmentation):** Eğitim setine döndürme, zoom ve kaydırma işlemleri uygulanarak modelin genelleme yeteneği maksimize edildi.
        """)
    with t2:
        st.write("**Mimari:** MobileNetV2 (Transfer Learning)")
        st.info("Düşük hesaplama gücü ile yüksek doğruluk sağladığı için medikal sistemlere en uygun model seçilmiştir.")
        st.table(pd.DataFrame({
            "Parametre": ["Optimizer", "Learning Rate", "Batch Size", "Loss Function"],
            "Değer": ["Adam", "0.0001", "32", "Categorical Cross-Entropy"]
        }))
    st.markdown('</div>', unsafe_allow_html=True)

# 📊 PERFORMANS ANALİZİ (İSTEDİĞİN ACCURACY BURADA)
elif page == "📊 Performans Analizi":
    st.header("📊 Bölüm 3: Akademik Başarı Metrikleri")
    
    # Metrik Kartları
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Genel Doğruluk", "%68")
    m2.metric("AUC Skoru", "0.94")
    m3.metric("F1-Skoru", "0.65")
    m4.metric("Test Kaybı", "0.52")

    st.divider()
    
    # Accuracy ve Loss Grafiği (Kriter 15-16)
    st.subheader("📈 Eğitim Süreci: Doğruluk ve Kayıp Analizi")
    if os.path.exists('learning_curves.png'):
        st.image('learning_curves.png', caption="Doğruluk (Accuracy) ve Kayıp (Loss) Eğrileri", use_container_width=True)
    st.markdown('<p class="academic-label"><b>Yorum:</b> Eğitim ve doğrulama eğrilerinin paralelliği, modelin overfitting yapmadan stabil bir öğrenme gerçekleştirdiğini kanıtlar.</p>', unsafe_allow_html=True)

    st.divider()

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("📍 Karmaşıklık Matrisi")
        if os.path.exists('kidney_confusion_matrix.png'):
            st.image('kidney_confusion_matrix.png', use_container_width=True)
        st.markdown('<p class="academic-label"><b>Analiz:</b> Sağlıklı dokuların ayırt edilmesinde %84 başarı sağlanmıştır.</p>', unsafe_allow_html=True)
    with col_g2:
        st.subheader("📉 ROC Eğrisi")
        if os.path.exists('kidney_roc_curve.png'):
            st.image('kidney_roc_curve.png', use_container_width=True)
        st.markdown('<p class="academic-label"><b>Analiz:</b> 0.94 AUC skoru yüksek ayırt ediciliği belgeler.</p>', unsafe_allow_html=True)

# 🔬 CANLI TANI LABORATUVARI
elif page == "🔬 Canlı Tanı Laboratuvarı":
    st.header("🔬 Bölüm 4: BT Kesiti Canlı Teşhis Merkezi")
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Lütfen bir BT görüntüsü yükleyin...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and model:
        c1, c2 = st.columns(2)
        img = Image.open(uploaded_file).convert('RGB')
        with c1:
            st.image(img, caption="Giriş Görüntüsü", use_container_width=True)
        with c2:
            if st.button("ANALİZİ BAŞLAT"):
                with st.spinner('Yapay Zeka İnceliyor...'):
                    # İşleme
                    p_img = np.array(img.resize((224, 224))).astype('float32') / 255.0
                    p_img = np.expand_dims(p_img, axis=0)
                    preds = model.predict(p_img, verbose=0)
                    idx = np.argmax(preds)
                    
                    st.subheader("Teşhis Çıkarımı")
                    res_color = "#238636" if idx == 1 else "#da3633"
                    st.markdown(f"<h1 style='color: {res_color};'>{LABELS[idx]}</h1>", unsafe_allow_html=True)
                    st.metric("Güven Oranı", f"%{np.max(preds)*100:.1f}")
                    
                    df_p = pd.DataFrame({'Patoloji': LABELS, 'Olasılık': preds[0]})
                    st.plotly_chart(px.bar(df_p, x='Patoloji', y='Olasılık', color='Patoloji', template="plotly_dark"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- ALT BİLGİ ---
st.divider()
st.markdown(f"<div style='text-align: center; color: #8b949e;'>Giresun Üniversitesi Bilgisayar Mühendisliği - {st.session_state.get('date', '2026')}</div>", unsafe_allow_html=True)
