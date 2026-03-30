import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json

# 1. SAYFA YAPILANDIRMASI (Madde 17 & 18: Navigasyon ve Estetik)
st.set_page_config(page_title="Renal AI | Böbrek Patolojisi Analizi", layout="wide")

# --- MEDİKAL TERMİNAL TEMASI (Özel CSS) ---
st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #d1d5db; }
    .main-card { background-color: #161b22; border: 1px solid #30363d; padding: 30px; border-radius: 12px; margin-bottom: 25px; }
    .metric-card { background-color: #1f2937; border-left: 5px solid #3b82f6; padding: 20px; border-radius: 8px; }
    h1, h2, h3 { color: #60a5fa !important; font-weight: 700; }
    .stButton>button { background-color: #2563eb; color: white; border-radius: 6px; width: 100%; border: none; height: 45px; font-weight: 600; }
    .stButton>button:hover { background-color: #1d4ed8; border: none; }
    .status-tag { background-color: #238636; color: white; padding: 4px 12px; border-radius: 20px; font-size: 14px; }
    </style>
    """, unsafe_allow_html=True)

# 2. MODEL YÜKLEME (Madde 19: Teknik Çalışırlık)
MODEL_PATH = 'kidney_disease_mobilenet_model2.h5'

@st.cache_resource
def load_kidney_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Kritik Hata: Model dosyası ({MODEL_PATH}) bulunamadı.")
        return None
    try:
        # Keras 3 Uyumluluk Yaması
        from tensorflow.keras.layers import InputLayer
        class CompatibleInputLayer(InputLayer):
            def __init__(self, *args, **kwargs):
                if 'batch_shape' in kwargs:
                    kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
                super().__init__(*args, **kwargs)
        
        return tf.keras.models.load_model(
            MODEL_PATH, 
            compile=False, 
            custom_objects={'InputLayer': CompatibleInputLayer}
        )
    except Exception as e:
        st.error(f"⚠️ Model Yapılandırma Hatası: {e}")
        return None

model = load_kidney_model()
LABELS = ['Cyst (Kist)', 'Normal', 'Stone (Taş)', 'Tumor (Tümör)']

# --- ÜST BİLGİ PANELİ ---
st.title("🧬 Renal AI: Klinik Karar Destek Sistemi")
st.markdown(f"**Araştırmacı:** Oğuzhan Dursun (**No:** 220706037) | **Kurum:** Giresun Üniversitesi")
st.divider()

# 3. AKADEMİK RAPOR AKIŞI (Puanlama Kriterleri 1-11)
st.header("📋 Proje Dokümantasyonu ve Metodoloji")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("1. Problem ve Amaç")
    st.write("""
    Böbrek kanseri ve kistlerinin BT görüntülerinden teşhisi yüksek dikkat gerektiren bir radyolojik süreçtir. 
    **Renal AI**, yapay zeka algoritmalarını kullanarak BT kesitlerini tarar ve potansiyel patolojileri (Kist, Taş, Tümör) 
    tespit ederek radyologlar için bir 'ikinci göz' işlevi görür.
    \n**Hedef:** Klinik hata payını minimize etmek ve teşhis hızını artırmaktır.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("2. Veri Seti ve Ön İşleme")
    st.write("""
    **Veri:** Kaggle CT-Kidney Dataset (12.000+ görüntü).
    \n**İşlemler:** Görüntüler 224x224 boyutuna normalize edilmiş, pikseller [0,1] aralığına ölçeklenmiştir. 
    Eğitimde **Data Augmentation** (Döndürme, Zoom) kullanılarak modelin genelleme yeteneği artırılmıştır.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# MODEL MİMARİSİ
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.subheader("3. Model Mimarisi ve Eğitim Protokolü")
c1, c2, c3 = st.columns(3)
c1.write("**Mimari:** MobileNetV2\n(Hızlı ve Düşük CPU Tüketimi)")
c2.write("**Parametreler:**\nAdam Opt., lr=0.0001")
c3.write("**Kayıp Fonksiyonu:**\nCategorical Crossentropy")
st.markdown('</div>', unsafe_allow_html=True)

# 4. PERFORMANS ANALİZİ (Madde 12-16)
st.divider()
st.header("📊 Akademik Performans Analizi")

m1, m2, m3, m4 = st.columns(4)
with m1: st.markdown('<div class="metric-card"><b>Doğruluk:</b><br><h2>%68</h2></div>', unsafe_allow_html=True)
with m2: st.markdown('<div class="metric-card"><b>F1-Score:</b><br><h2>0.65</h2></div>', unsafe_allow_html=True)
with m3: st.markdown('<div class="metric-card"><b>AUC Skoru:</b><br><h2>0.94</h2></div>', unsafe_allow_html=True)
with m4: st.markdown('<div class="metric-card"><b>Normal F1:</b><br><h2>0.84</h2></div>', unsafe_allow_html=True)

st.write("---")
g1, g2 = st.columns(2)

with g1:
    st.subheader("📍 Karmaşıklık Matrisi")
    if os.path.exists('kidney_confusion_matrix.png'):
        st.image('kidney_confusion_matrix.png', width=600)
    st.markdown('<p class="academic-note"><b>Yorum:</b> Modelin sağlıklı (Normal) dokuyu ayırt etme başarısı %84 ile oldukça yüksektir. Taş ve Tümör sınıfları arasındaki görsel benzerlikler kısıtlı karışıklığa neden olmaktadır.</p>', unsafe_allow_html=True)

with g2:
    st.subheader("📉 ROC/AUC Analizi")
    if os.path.exists('kidney_roc_curve.png'):
        st.image('kidney_roc_curve.png', width=600)
    st.markdown('<p class="academic-note"><b>Yorum:</b> AUC değerinin 0.94 olması, modelin rastgele bir tahminden çok daha üstün, yüksek bir ayırt edicilik gücüne sahip olduğunu bilimsel olarak kanıtlar.</p>', unsafe_allow_html=True)

# 5. CANLI ANALİZ LABORATUVARI (Madde 19: Teknik Çalışırlık)
st.divider()
st.header("🔬 BT Kesiti Canlı Analiz Merkezi")
st.write("Lütfen sisteme bir BT (Computed Tomography) görüntüsü yükleyiniz.")

up_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if up_file and model:
    img = Image.open(up_file).convert('RGB')
    
    col_img, col_res = st.columns(2)
    with col_img:
        st.image(img, caption="Giriş BT Kesiti", use_container_width=True)
    
    with col_res:
        if st.button("ANALİZİ BAŞLAT"):
            with st.spinner('Yapay Zeka İnceliyor...'):
                # İşleme
                p_img = np.array(img.resize((224, 224))).astype('float32') / 255.0
                p_img = np.expand_dims(p_img, axis=0)
                preds = model.predict(p_img, verbose=0)
                idx = np.argmax(preds)
                
                st.subheader("Tahhis Çıkarımı")
                res_color = "#238636" if idx == 1 else "#da3633"
                st.markdown(f"<h1 style='color: {res_color};'>{LABELS[idx]}</h1>", unsafe_allow_html=True)
                st.metric("Tespit Güveni", f"%{np.max(preds)*100:.1f}")
                
                # Grafik
                df_res = pd.DataFrame({'Sınıf': LABELS, 'Olasılık': preds[0]})
                fig = px.bar(df_res, x='Sınıf', y='Olasılık', color='Sınıf', template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

# SONUÇ VE KAYNAKÇA (Madde 20)
st.divider()
st.subheader("📚 Sonuç ve Kaynakça")
st.write("""
Bu çalışma kapsamında geliştirilen sistem, sağlık bilişiminde derin öğrenme kullanımına örnektir. 
\n**Kaynakça:** 1. Kaggle CT-Kidney Dataset. 
2. MobileNetV2: Sandler et al. (2018). 
3. TensorFlow Keras API Documentation.
""")
st.caption("Oğuzhan Dursun - 220706037 | © 2026 Sağlık Bilişimi")
