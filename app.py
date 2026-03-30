import streamlit as st            # Web tabanlı kullanıcı arayüzünü (UI) oluşturmak için temel kütüphane.
import tensorflow as tf           # Derin öğrenme modelini (Keras) arka planda çalıştırmak için.
from PIL import Image             # Kullanıcının yüklediği görüntü dosyalarını açmak ve işlemek için.
import numpy as np                # Görüntüleri sayısal dizilere (matris) çevirmek ve normalizasyon için.
import pandas as pd               # Tahmin olasılıklarını tablo yapısına dönüştürüp grafiğe hazırlamak için.
import plotly.express as px       # Tahmin sonuçlarını interaktif sütun grafiklerine dönüştürmek için.
import os                         # Dosya yolları, dizin kontrolü ve dosya varlık sorgulaması için.

# 1. SAYFA YAPILANDIRMASI (Madde 17, 18 & 19: Estetik ve Teknik Çalışırlık)
st.set_page_config(page_title="Böbrek Analiz Pro", layout="wide") # Sayfa başlığını ve geniş ekran modunu ayarlar.

# --- GELİŞMİŞ TIBBİ PANEL CSS (Görsel Tasarım ve Kullanılabilirlik) ---
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; } /* Uygulama arka plan rengini koyu tema yapar. */
    div[data-testid="stHorizontalBlock"] {
        background-color: #161b22;
        padding: 10px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 30px;
    } /* Navigasyon barı için özel çerçeve ve gölge ayarları. */
    .medical-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
    } /* İçeriklerin içine yerleştiği tıbbi kart tasarımı. */
    .medical-card h1, .medical-card h2, .medical-card h3 {
        margin-top: 0px !important;
        padding-top: 0px !important;
    } /* Kart başlıklarının üst boşluklarını sıfırlayarak hizalamayı düzeltir. */
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Inter', sans-serif; font-weight: 700; } /* Başlık renkleri. */
    .stButton>button {
        background: linear-gradient(90deg, #1d4ed8, #2563eb);
        color: white;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        height: 55px;
        width: 100%;
        transition: 0.4s;
    } /* Modern ve degrade geçişli buton tasarımı. */
    .stButton>button:hover { transform: scale(1.01); box-shadow: 0 5px 15px rgba(88, 166, 255, 0.4); } /* Buton hover efekti. */
    </style>
    """, unsafe_allow_html=True) # HTML ve CSS kodlarını uygulamaya enjekte eder.

# 2. MODEL YÜKLEME VE VERSİYON UYUMLULUK ÇÖZÜMÜ (Madde 9: Model Mimarisi)
@st.cache_resource # Modelin bellekte tutulmasını sağlayarak performansı artırır.
def load_trained_model():
    model_path = 'kidney_disease_mobilenet_model2.h5' # Eğitilmiş model dosyasının yolu.
    if not os.path.exists(model_path): # Dosya kontrolü yapar.
        st.error(f"❌ Model dosyası bulunamadı!") # Dosya yoksa uyarı verir.
        return None # Boş döner.

    from tensorflow.keras.layers import InputLayer # Keras katman sınıfını içe aktarır.
    class CompatibleInputLayer(InputLayer): # Yeni Keras sürümleri için uyumluluk katmanı.
        def __init__(self, *args, **kwargs): # Yapıcı metod.
            if 'batch_shape' in kwargs: # Eski parametre ismini kontrol eder.
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape') # Yeni isme dönüştürür.
            super().__init__(*args, **kwargs) # Üst sınıfı başlatır.

    try:
        return tf.keras.models.load_model(
            model_path, 
            compile=False, 
            custom_objects={'InputLayer': CompatibleInputLayer}
        ) # Modeli özel katman tanımıyla beraber yükler.
    except Exception as e: # Hata durumunda:
        st.error(f"⚠️ Model Yükleme Hatası: {e}") # Hatayı gösterir.
        return None # Boş döner.

model = load_trained_model() # Fonksiyonu çağırarak modeli yükler.
LABELS = ['Cyst (Kist)', 'Normal', 'Stone (Taş)', 'Tumor (Tümör)'] # Sınıf etiketleri.

# 3. ÜST NAVBAR (Madde 18: Gezinme Kolaylığı)
st.title("🛡️ Böbrek Analiz Pro: İleri Seviye Böbrek Analiz Sistemi") # Uygulama başlığı.
menu_col1, menu_col2, menu_col3, menu_col4 = st.columns(4) # Menü butonları için 4 sütun.

with menu_col1: # Proje Vizyonu butonu.
    if st.button("🏠 PROJE VİZYONU"): st.session_state.page = "vizyon" # Sayfa durumunu günceller.
with menu_col2: # Teknik Altyapı butonu.
    if st.button("🧠 TEKNİK ALTYAPI"): st.session_state.page = "teknik" # Sayfa durumunu günceller.
with menu_col3: # Analitik Raporlar butonu.
    if st.button("📊 ANALİTİK RAPORLAR"): st.session_state.page = "analitik" # Sayfa durumunu günceller.
with menu_col4: # Canlı Tanı butonu.
    if st.button("🔬 CANLI TANI MERKEZİ"): st.session_state.page = "tani" # Sayfa durumunu günceller.

if 'page' not in st.session_state: st.session_state.page = "vizyon" # Varsayılan sayfayı belirler.
st.divider() # Görsel ayırıcı.

# --- BÖLÜM 1: VİZYON (Madde 1, 2, 3: Problem ve Amaç) ---
if st.session_state.page == "vizyon": # Vizyon sayfası seçiliyse:
    st.header("🏥 Bölüm 1: Problemin Tanımı ve Akademik Önemi") # Başlık.
    st.markdown('<div class="medical-card">', unsafe_allow_html=True) # Kart başlatır.
    c1, c2 = st.columns([2, 1]) # İçeriği böler.
    with c1: # Metin alanı.
        st.subheader("📌 Problemin Tanımı ve Önemi") # Alt başlık.
        st.write("""
        BT görüntülerindeki patolojilerin manuel taranması tıbbi hata riskini artırır. 
        **Amacımız:** Hekimlerin karar süreçlerini hızlandıran bir karar destek mekanizması sunmaktır.
        """) # Problem tanımı metni.
    with c2: # Veri seti alanı.
        st.subheader("📚 Veri Seti (Madde 4 & 5)") # Alt başlık.
        st.info("Kaynak: Kaggle CT-Kidney | Sınıf Sayısı: 4") # Veri kaynağı bilgisi.
    st.markdown('</div>', unsafe_allow_html=True) # Kart kapatır.

# --- BÖLÜM 2: TEKNİK ALTYAPI (Madde 6, 7, 8, 10, 11: Metodoloji ve Eğitim) ---
elif st.session_state.page == "teknik": # Teknik sayfa seçiliyse:
    st.header("🧬 Bölüm 2: Metodoloji ve Eğitim Süreci") # Başlık.
    st.markdown('<div class="medical-card">', unsafe_allow_html=True) # Kart başlatır.
    col_arch, col_params = st.columns(2) # Sütunlara böler.
    with col_arch: # Mimari açıklaması.
        st.subheader("🧠 Mimari ve Ön İşleme") # Alt başlık.
        st.write("- Model: MobileNetV2 | Ayrım: %80 Eğitim, %20 Doğrulama") # Teknik detaylar.
    with col_params: # Hiperparametre açıklaması.
        st.subheader("⚙️ Hiperparametreler (Madde 10)") # Alt başlık.
        st.success("- Optimizer: Adam (LR: 0.0001) | Epoch: 25 | Batch: 32") # Parametre listesi.
    st.divider() # Ayırıcı.
    st.subheader("📈 Eğitim Sürecinin Açıklanması (Madde 11)") # Eğitim süreci başlığı.
    st.write("Overfitting'i engellemek için Early Stopping mekanizması kullanılmıştır.") # Teknik açıklama.
    st.markdown('</div>', unsafe_allow_html=True) # Kart kapatır.

# --- BÖLÜM 3: ANALİTİK RAPORLAR (Madde 12, 13, 14, 15, 16: Metrikler ve Grafikler) ---
elif st.session_state.page == "analitik": # Analitik sayfa seçiliyse:
    st.header("📊 Bölüm 3: Performans Analizi ve Grafik Yorumları") # Başlık.
    m1, m2, m3, m4 = st.columns(4) # Metrikler için 4 sütun.
    m1.metric("Doğruluk (Acc)", "%68") # Doğruluk değeri.
    m2.metric("AUC Skoru", "0.94") # AUC değeri.
    m3.metric("F1-Skoru", "0.65") # F1 değeri.
    m4.metric("Duyarlılık", "0.88") # Duyarlılık değeri.
    st.divider() # Ayırıcı.
    if os.path.exists('learning_curves.png'): # Grafik dosyasını kontrol eder.
        st.image('learning_curves.png', caption="Şekil 1: Eğitim Grafikleri", use_container_width=True) # Grafiği basar.
        st.info("**Yorum (Madde 14):** AUC 0.94 olması sınıfların başarılı ayrıştırıldığını kanıtlarlar.") # Teknik yorum.
    g1, g2 = st.columns(2) # Alt grafikler için 2 sütun.
    with g1: # Hata matrisi.
        if os.path.exists('kidney_confusion_matrix.png'): st.image('kidney_confusion_matrix.png', use_container_width=True) # CM görseli.
    with g2: # ROC eğrisi.
        if os.path.exists('kidney_roc_curve.png'): st.image('kidney_roc_curve.png', use_container_width=True) # ROC görseli.

# --- BÖLÜM 4: CANLI TANI MERKEZİ ---
elif st.session_state.page == "tani": # Tanı sayfası seçiliyse:
    st.header("🔬 Bölüm 4: BT Kesiti Canlı Analiz Modülü") # Başlık.
    st.markdown('<div class="medical-card">', unsafe_allow_html=True) # Kart başlatır.
    up_file = st.file_uploader("Görüntü yükleyin...", type=["jpg", "png", "jpeg"]) # Dosya yükleyici.
    if up_file: # Dosya yüklendiyse:
        col_img, col_analysis = st.columns([1, 1], gap="large") # Görsel ve analiz sütunları.
        img = Image.open(up_file).convert('RGB') # Resmi açar.
        with col_img: st.image(img, caption="Yüklenen Kesit", width=420) # Resmi gösterir.
        with col_analysis: # Analiz alanı.
            if model is not None: # Model yüklüyse:
                if st.button("🚀 ANALİZİ BAŞLAT"): # Butona basıldıysa:
                    with st.spinner('Analiz ediliyor...'): # Yükleme efekti.
                        p_img = np.array(img.resize((224, 224))).astype('float32') / 255.0 # Ön işleme.
                        preds = model.predict(np.expand_dims(p_img, axis=0), verbose=0) # Tahmin.
                        idx = np.argmax(preds) # En yüksek sınıf.
                        st.markdown(f"### Teşhis: {LABELS[idx]}") # Sonucu basar.
                        st.plotly_chart(px.bar(x=LABELS, y=preds[0], template="plotly_dark"), use_container_width=True) # Grafik.
    st.markdown('</div>', unsafe_allow_html=True) # Kart kapatır.

# 5. SONUÇ VE KAYNAKÇA (Madde 20: Genel Bütünlük)
st.divider() # Ayırıcı.
st.subheader("🏁 Sonuç ve Kaynakça") # Kapanış başlığı.
st.write("Bu çalışma derin öğrenmenin teşhis süreçlerindeki gücünü valide etmektedir.") # Sonuç metni.
st.markdown("<div style='text-align: center; color: #8b949e;'><b>Geliştirici:</b> Oğuzhan Dursun - 220706037 | <b>Ders:</b> Sağlık Bilişimi 2026</div>", unsafe_allow_html=True) # İmza alanı.
