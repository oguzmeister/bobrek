import streamlit as st           # Web arayüzü oluşturmak için temel kütüphane.
import tensorflow as tf          # Derin öğrenme modelini (Keras) çalıştırmak için.
from PIL import Image            # Görüntü dosyalarını açmak ve boyutlandırmak için.
import numpy as np               # Sayısal matris işlemleri ve normalizasyon için.
import pandas as pd              # Tahmin sonuçlarını tablo yapısına dönüştürmek için.
import plotly.express as px      # İnteraktif grafikler (Bar chart) oluşturmak için.
import os                       # Dosya yolları ve sistem kontrolleri için.

# 1. SAYFA YAPILANDIRMASI (Madde 17 & 18: Estetik ve Gezinme)
# st.set_page_config: Tarayıcı sekme başlığını ve sayfa yerleşimini ayarlar.
st.set_page_config(page_title="Renal AI | Böbrek Analiz Pro", layout="wide")

# --- MEDİKAL TEMA CSS (Görsel Arayüz Tasarımı) ---
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; } /* Koyu arka plan */
    /* Üst Navbar (Menü) Şeridinin Tasarımı */
    div[data-testid="stHorizontalBlock"] {
        background-color: #161b22;
        padding: 10px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 30px;
    }
    /* Bilgi ve Rapor Kartlarının Tasarımı */
    .medical-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    /* Başlıkların Medikal Mavi Tonlarına Ayarlanması */
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Inter', sans-serif; font-weight: 700; }
    /* Analiz Butonu İçin Degrade (Gradient) Tasarım */
    .stButton>button {
        background: linear-gradient(90deg, #1d4ed8, #2563eb);
        color: white;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        height: 55px;
        width: 100%;
        transition: 0.4s;
    }
    /* Buton Üzerine Gelindiğinde Parlama Efekti */
    .stButton>button:hover { transform: scale(1.01); box-shadow: 0 5px 15px rgba(88, 166, 255, 0.4); }
    </style>
    """, unsafe_allow_html=True)

# 2. MODEL YÜKLEME VE VERSİYON UYUMLULUK ÇÖZÜMÜ (Madde 19)
@st.cache_resource # Modeli belleğe alır, her işlemde tekrar yüklenmesini önler.
def load_trained_model():
    model_path = 'kidney_disease_mobilenet_model2.h5' # Ana dizindeki model dosyası.
    
    if not os.path.exists(model_path): # Dosya sistemde var mı kontrolü.
        st.error(f"❌ Model dosyası ({model_path}) bulunamadı!")
        return None

    # --- KERAS 3 UYUMLULUK YAMASI (GÜVENLİ YÖNTEM) ---
    # Bu sınıf, Keras 2'deki 'batch_shape' parametresini Keras 3'ün beklediği 'batch_input_shape'e yükleme anında çevirir.
    from tensorflow.keras.layers import InputLayer
    class CompatibleInputLayer(InputLayer):
        def __init__(self, *args, **kwargs):
            # Eğer yapılandırma içinde eski 'batch_shape' varsa onu yeni anahtara aktarır.
            if 'batch_shape' in kwargs:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
            super().__init__(*args, **kwargs)

    try:
        # compile=False: Eğitim parametrelerini yüklemeyerek sürüm çakışmalarını önler.
        # custom_objects: Keras'a giriş katmanını bizim hazırladığımız uyumlu sınıfla okumasını söyler.
        return tf.keras.models.load_model(
            model_path, 
            compile=False, 
            custom_objects={'InputLayer': CompatibleInputLayer}
        )
    except Exception as e:
        st.error(f"⚠️ Model Yükleme Hatası: {e}")
        return None

# Modeli belleğe çağırır ve sınıfları tanımlar.
model = load_trained_model()
LABELS = ['Cyst (Kist)', 'Normal', 'Stone (Taş)', 'Tumor (Tümör)']

# 3. ÜST NAVBAR (Yatay Navigasyon Menüsü)
st.title("🛡️ Renal AI: İleri Seviye Böbrek Analiz Sistemi")
menu_col1, menu_col2, menu_col3, menu_col4 = st.columns(4) # Menü için 4 eşit sütun.

# Butonlara basıldığında 'session_state' üzerinden sayfa değişimi tetiklenir.
with menu_col1:
    if st.button("🏠 PROJE VİZYONU"): st.session_state.page = "vizyon"
with menu_col2:
    if st.button("🧠 TEKNİK ALTYAPI"): st.session_state.page = "teknik"
with menu_col3:
    if st.button("📊 ANALİTİK RAPORLAR"): st.session_state.page = "analitik"
with menu_col4:
    if st.button("🔬 CANLI TANI MERKEZİ"): st.session_state.page = "tani"

if 'page' not in st.session_state: st.session_state.page = "vizyon" # Varsayılan açılış sayfası.
st.divider()

# --- BÖLÜM 1: VİZYON ---
if st.session_state.page == "vizyon":
    st.header("🏥 Bölüm 1: Problemin Tanımı ve Akademik Önemi")
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📌 Problemin Tanımı")
        st.write("Böbrek patolojilerinin BT kesitlerinden teşhisi radyologlar için zaman alıcıdır. Renal AI, karar destek mekanizması olarak teşhis doğruluğunu artırır.")
    with c2:
        st.subheader("📚 Veri Seti")
        st.info("**Kaynak:** Kaggle CT-Kidney\n\n**Sınıf:** 4 (Normal, Kist, Taş, Tümör)")
    st.markdown('</div>', unsafe_allow_html=True)

# --- BÖLÜM 2: TEKNİK ALTYAPI ---
elif st.session_state.page == "teknik":
    st.header("🧬 Bölüm 2: Mühendislik ve Model Altyapısı")
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    st.write("- **Model:** MobileNetV2 (Transfer Learning)\n- **Ön İşleme:** 1/255 Normalizasyon & 224x224 Resize.\n- **Optimizasyon:** Adam Optimizer (lr=0.0001).")
    st.markdown('</div>', unsafe_allow_html=True)

# --- BÖLÜM 3: ANALİTİK RAPORLAR ---
elif st.session_state.page == "analitik":
    st.header("📊 Bölüm 3: Model Başarım Metrikleri")
    m1, m2, m3, m4 = st.columns(4) # Puanlama kriteri olan 4 temel metriği basar.
    m1.metric("Doğruluk (Acc)", "%68")
    m2.metric("AUC Skoru", "0.94")
    m3.metric("F1-Skoru", "0.65")
    m4.metric("Duyarlılık", "0.88")
    st.divider()
    # Accuracy ve Loss grafiği (Madde 15).
    if os.path.exists('learning_curves.png'): st.image('learning_curves.png', caption="Eğitim Grafikleri", width=1100)
    g1, g2 = st.columns(2)
    with g1:
        if os.path.exists('kidney_confusion_matrix.png'): st.image('kidney_confusion_matrix.png', caption="Karmaşıklık Matrisi", use_container_width=True)
    with g2:
        if os.path.exists('kidney_roc_curve.png'): st.image('kidney_roc_curve.png', caption="ROC Analizi", use_container_width=True)

# --- BÖLÜM 4: CANLI TANI MERKEZİ (HİZALAMA DÜZELTİLDİ) ---
elif st.session_state.page == "tani":
    st.header("🔬 Bölüm 4: BT Kesiti Canlı Analiz Modülü")
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    up_file = st.file_uploader("Analiz edilecek BT görüntüsünü yükleyiniz...", type=["jpg", "png", "jpeg"])
    
    if up_file:
        col_img, col_analysis = st.columns([1, 1], gap="large") # Görsel ve analizi yan yana hizalar.
        img = Image.open(up_file).convert('RGB') # Resmi RGB kanalına çevirir.
        
        with col_img:
            st.markdown("### 🖼️ Giriş BT Kesiti")
            st.image(img, caption="Yüklenen Ham Görüntü", width=420) # Genişliği sabitleyerek kaymayı önler.
        
        with col_analysis:
            st.markdown("### 🤖 Analiz Sonuçları")
            if model is not None:
                if st.button("🚀 ANALİZİ BAŞLAT"): # Kullanıcı butona basınca tahmin başlar.
                    with st.spinner('Yapay Zeka Taraması Yapılıyor...'):
                        # Görüntü hazırlama: 224x224 boyut ve 0-1 arası normalizasyon.
                        p_img = np.array(img.resize((224, 224))).astype('float32') / 255.0
                        p_img = np.expand_dims(p_img, axis=0) # Batch boyutu ekleme.
                        preds = model.predict(p_img, verbose=0) # Tahmin dizisi üretme.
                        idx = np.argmax(preds)                 # En yüksek sınıfı bulma.
                        
                        # Teşhis Kartı: Sonuca göre dinamik renk değişimi.
                        res_color = "#238636" if idx == 1 else "#da3633"
                        st.markdown(f"""
                            <div style="border-left: 10px solid {res_color}; padding: 20px; background-color: #1f2937; border-radius: 12px; margin-top:0;">
                                <h2 style="margin:0; color: white !important;">Teşhis: {LABELS[idx]}</h2>
                                <h3 style="margin:0; color: #8b949e !important;">Güven Oranı: %{np.max(preds)*100:.1f}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Plotly Grafiği: Olasılıkları tablo (DataFrame) yapıp bar grafiğe döker.
                        df_p = pd.DataFrame({'Patoloji': LABELS, 'Olasılık': preds[0]})
                        fig = px.bar(df_p, x='Patoloji', y='Olasılık', color='Patoloji', height=300, template="plotly_dark")
                        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=10, b=10))
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Model yüklenemedi. Lütfen dosyayı kontrol edin.")
    st.markdown('</div>', unsafe_allow_html=True)

# 4. ALT BİLGİ (Madde 20: Akademik Kimlik)
st.divider()
st.markdown(f"<div style='text-align: center; color: #8b949e;'><b>Geliştirici:</b> Oğuzhan Dursun - 220706037 | <b>Ders:</b> Yapay Zeka ile Sağlık Bilişimi Vize Ödevi - 2026</div>", unsafe_allow_html=True)
