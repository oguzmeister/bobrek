import streamlit as st           # Web arayüzü oluşturmak için temel kütüphane.
import tensorflow as tf          # Derin öğrenme modelini çalıştırmak için Keras motoru.
from PIL import Image            # Görüntü dosyalarını açmak ve işlemek için kütüphane.
import numpy as np               # Sayısal matris işlemleri ve veri normalizasyonu için.
import pandas as pd              # Verileri tablo (DataFrame) yapısında düzenlemek için.
import plotly.express as px      # İnteraktif grafikler (Bar chart vb.) oluşturmak için.
import plotly.graph_objects as go # Özelleştirilmiş grafik çizimleri (ROC eğrisi vb.) için.
import os                       # Dosya yolları ve dizin kontrolleri için sistem kütüphanesi.
import h5py                     # .h5 model dosyalarının içine düşük seviyeli erişim için.
import json                     # Model yapılandırmasını (config) okumak ve düzenlemek için.

# 1. SAYFA YAPILANDIRMASI (Madde 17 & 18: Estetik ve Gezinme)
# st.set_page_config: Tarayıcı sekme başlığını ve sayfa genişliğini ayarlar.
st.set_page_config(page_title="Renal AI | Böbrek Analiz Pro", layout="wide")

# st.markdown(unsafe_allow_html=True): Sayfaya özel CSS enjekte ederek medikal tema oluşturur.
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; } /* Koyu tema arka planı */
    div[data-testid="stHorizontalBlock"] { /* Üst navbar şeridinin arka plan ve çerçevesi */
        background-color: #161b22;
        padding: 10px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 30px;
    }
    .medical-card { /* Bilgi blokları (kartlar) için özel tasarım */
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Inter', sans-serif; font-weight: 700; } /* Başlık renkleri */
    .stButton>button { /* Navigasyon ve analiz butonları için mavi degrade (gradient) stili */
        background: linear-gradient(90deg, #1d4ed8, #2563eb);
        color: white;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        height: 55px;
        width: 100%;
        transition: 0.4s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 5px 15px rgba(88, 166, 255, 0.4); } /* Hover efekti */
    .academic-note { font-size: 14px; color: #8b949e; border-left: 4px solid #58a6ff; padding-left: 15px; font-style: italic; }
    </style>
    """, unsafe_allow_html=True)

# 2. MODEL YÜKLEME VE VERSİYON UYUMLULUK ÇÖZÜMÜ (Madde 19)
@st.cache_resource # Modeli bellekte saklar, her sayfa yenilemede tekrar yüklenmesini önler.
def load_trained_model():
    current_dir = os.path.dirname(os.path.abspath(__file__)) # Çalışan dosyanın tam dizinini bulur.
    model_name = 'kidney_disease_mobilenet_model2.h5'         # Model dosyasının adı.
    model_path = os.path.join(current_dir, model_name)       # Dizin ve dosya adını birleştirir.
    
    if not os.path.exists(model_path): # Dosya sistemde var mı kontrol eder.
        st.error(f"❌ Model dosyası bulunamadı: {model_name}")
        return None

    # KERAS 3 UYUMLULUK YAMASI: Eski sürüm modellerdeki 'batch_shape' hatasını yüklemeden önce onarır.
    try:
        with h5py.File(model_path, 'r+') as f: # Dosyayı okuma-yazma modunda açar.
            if 'model_config' in f.attrs: # Model yapılandırma meta verisini bulur.
                config_raw = f.attrs['model_config'] 
                if isinstance(config_raw, bytes): config_raw = config_raw.decode('utf-8') # Byte veriyi metne çevirir.
                config_dict = json.loads(config_raw) # Metni Python sözlüğüne (dictionary) çevirir.
                modified = False
                for layer in config_dict['config']['layers']: # Tüm katmanları tarar.
                    if 'batch_shape' in layer['config']: # Eski parametreyi bulursa;
                        layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape') # Yeni parametreyle değiştirir.
                        modified = True
                if modified: f.attrs['model_config'] = json.dumps(config_dict).encode('utf-8') # Güncel halini geri yazar.
    except: pass # Dosya salt okunur ise veya hata oluşursa devam eder.

    # Özel InputLayer tanımı: Keras yükleme anında hata vermemesi için uyumlu katman sınıfı.
    from tensorflow.keras.layers import InputLayer
    class CompatibleInputLayer(InputLayer):
        def __init__(self, *args, **kwargs):
            if 'batch_shape' in kwargs: kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
            super().__init__(*args, **kwargs)

    try:
        # Modeli yüklerken özel katmanı sisteme tanıtır (custom_objects).
        return tf.keras.models.load_model(model_path, compile=False, custom_objects={'InputLayer': CompatibleInputLayer})
    except Exception as e:
        st.error(f"Keras Yükleme Hatası: {e}")
        return None

# Modeli belleğe yükler ve sınıf isimlerini tanımlar.
model = load_trained_model()
LABELS = ['Cyst (Kist)', 'Normal', 'Stone (Taş)', 'Tumor (Tümör)']

# 3. ÜST NAVBAR (Yatay Menü Sistemi)
st.title("🛡️ Renal AI: İleri Seviye Böbrek Analiz Projesi")
menu_col1, menu_col2, menu_col3, menu_col4 = st.columns(4) # Sayfanın üstüne 4 eşit sütun açar.

# Butonlara tıklandığında session_state (oturum durumu) güncellenerek sayfa içeriği değişir.
with menu_col1:
    if st.button("🏠 PROJE VİZYONU"): st.session_state.page = "vizyon"
with menu_col2:
    if st.button("🧠 TEKNİK ALTYAPI"): st.session_state.page = "teknik"
with menu_col3:
    if st.button("📊 ANALİTİK RAPORLAR"): st.session_state.page = "analitik"
with menu_col4:
    if st.button("🔬 CANLI TANI MERKEZİ"): st.session_state.page = "tani"

if 'page' not in st.session_state: st.session_state.page = "vizyon" # Varsayılan açılış sayfası.
st.divider() # Görsel ayırıcı çizgi.

# --- BÖLÜM 1: PROJE VİZYONU ---
if st.session_state.page == "vizyon":
    st.header("🏥 Bölüm 1: Problemin Tanımı ve Akademik Önemi")
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1]) # İçerik için 2:1 oranında sütunlar.
    with c1:
        st.subheader("📌 Problemin Tanımı")
        st.write("Böbrek patolojilerinin BT kesitlerinden teşhisi, radyologlar için yüksek konsantrasyon gerektiren bir süreçtir. Renal AI, derin öğrenme temelli bir sistem geliştirerek radyolojik görüntülemede klinik karar destek mekanizması oluşturmayı hedefler.")
    with c2:
        st.subheader("📚 Veri Seti")
        st.info("**Kaynak:** Kaggle CT-Kidney\n\n**Sınıflar:** 4 (Kist, Normal, Taş, Tümör)\n\n**Kapsam:** 12.000+ Görüntü")
    st.markdown('</div>', unsafe_allow_html=True)

# --- BÖLÜM 2: TEKNİK ALTYAPI ---
elif st.session_state.page == "teknik":
    st.header("🧬 Bölüm 2: Mühendislik Yaklaşımı ve Mimari")
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    col_pre, col_arch = st.columns(2)
    with col_pre:
        st.subheader("⚙️ Veri Ön İşleme")
        st.write("- **Normalizasyon:** Piksel yoğunlukları [0, 1] aralığına ölçeklendi.\n- **Boyutlandırma:** Görüntüler 224x224 piksel formuna getirildi.\n- **Augmentation:** Döndürme ve Zoom işlemleriyle veri çeşitliliği artırıldı.")
    with col_arch:
        st.subheader("🧠 Model Mimarisi")
        st.write("- **Ana Model:** MobileNetV2 (Hızlı ve optimize mimari).\n- **Optimizer:** Adam (lr=0.0001).\n- **Loss:** Categorical Cross-Entropy.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- BÖLÜM 3: ANALİTİK RAPORLAR (Metrikler ve Grafikler) ---
elif st.session_state.page == "analitik":
    st.header("📊 Bölüm 3: Akademik Başarı Metrikleri")
    m1, m2, m3, m4 = st.columns(4) # Sayısal başarı metrikleri için 4 sütun.
    m1.metric("Genel Doğruluk", "%68")
    m2.metric("AUC Skoru", "0.94")
    m3.metric("F1-Skoru", "0.65")
    m4.metric("Duyarlılık", "0.88")
    st.divider()
    
    st.subheader("📈 Eğitim Süreci: Doğruluk ve Kayıp Analizi")
    if os.path.exists('learning_curves.png'): # Grafik dosyasını arar ve varsa ekrana basar.
        st.image('learning_curves.png', caption="Doğruluk ve Kayıp Eğrileri", width=1100)
    
    g1, g2 = st.columns(2) # Matris ve ROC eğrisi için yan yana iki sütun.
    with g1:
        if os.path.exists('kidney_confusion_matrix.png'): st.image('kidney_confusion_matrix.png', caption="Karmaşıklık Matrisi", use_container_width=True)
    with g2:
        if os.path.exists('kidney_roc_curve.png'): st.image('kidney_roc_curve.png', caption="ROC Analizi", use_container_width=True)

# --- BÖLÜM 4: CANLI TANI MERKEZİ (Yan Yana Hizalamalı) ---
elif st.session_state.page == "tani":
    st.header("🔬 Bölüm 4: BT Kesiti Canlı Analiz Modülü")
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    
    # Kullanıcının görüntü yüklemesini sağlar.
    up_file = st.file_uploader("Lütfen bir BT görüntüsü (JPG/PNG) yükleyiniz...", type=["jpg", "png", "jpeg"])
    
    if up_file:
        col_img, col_analysis = st.columns([1, 1], gap="large") # Resim ve analizi yan yana koyar.
        img = Image.open(up_file).convert('RGB') # Yüklenen dosyayı açar ve RGB kanalına çevirir.
        
        with col_img:
            st.markdown("### 🖼️ Giriş BT Kesiti")
            st.image(img, caption="Analiz Edilecek Görüntü", width=420) # Genişliği sabitleyerek kaymayı önler.
        
        with col_analysis:
            st.markdown("### 🤖 Analiz Sonuçları")
            if model is not None:
                if st.button("🚀 ANALİZİ BAŞLAT"): # Butona basıldığında tahmin motoru çalışır.
                    with st.spinner('Yapay Zeka İnceliyor...'):
                        # Görüntü hazırlama: 224x224 boyut ve 0-1 arası normalizasyon.
                        p_img = np.array(img.resize((224, 224))).astype('float32') / 255.0
                        p_img = np.expand_dims(p_img, axis=0) # Tekli görüntü için boyut artırımı (batch boyutu).
                        
                        preds = model.predict(p_img, verbose=0) # Model tahmini (olasılık dizisi döner).
                        idx = np.argmax(preds)                 # En yüksek olasılıklı sınıfın yerini bulur.
                        
                        # Sonuç kartı oluşturma (Normal ise yeşil, değilse kırmızı kenarlık).
                        res_color = "#238636" if idx == 1 else "#da3633"
                        st.markdown(f"""
                            <div style="border-left: 10px solid {res_color}; padding: 20px; background-color: #1f2937; border-radius: 12px; margin-top:0;">
                                <h2 style="margin:0; color: white !important; font-size: 26px;">Teşhis: {LABELS[idx]}</h2>
                                <h3 style="margin:0; color: #8b949e !important; font-size: 18px;">Güven Oranı: %{np.max(preds)*100:.1f}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Grafik Verisi Hazırlama: Olasılıkları tablo (DataFrame) haline getirir.
                        df_p = pd.DataFrame({'Patoloji': LABELS, 'Olasılık': preds[0]})
                        # İnteraktif Bar Grafik: Sınıf olasılıklarını görselleştirir.
                        fig = px.bar(df_p, x='Patoloji', y='Olasılık', color='Patoloji', height=300,
                                     template="plotly_dark", color_discrete_sequence=px.colors.sequential.Blues_r)
                        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=10, b=10))
                        st.plotly_chart(fig, use_container_width=True) # Grafiği sayfaya basar.
            else:
                st.error("Model yüklenemediği için analiz yapılamıyor.")
    st.markdown('</div>', unsafe_allow_html=True)

# 5. ALT BİLGİ (Akademik Künye)
st.divider()
st.markdown(f"""
    <div style='text-align: center; color: #8b949e; padding-bottom: 20px;'>
    <b>Geliştirici:</b> Oğuzhan Dursun - 220706037 | <b>Ders:</b> Yapay Zeka ile Sağlık Bilişimi Vize Ödevi - 2026
    </div>
    """, unsafe_allow_html=True)
