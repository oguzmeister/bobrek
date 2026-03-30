import streamlit as st           # Web tabanlı kullanıcı arayüzünü (UI) oluşturmak için temel kütüphane.
import tensorflow as tf          # Derin öğrenme modelini (Keras) arka planda çalıştırmak için.
from PIL import Image            # Kullanıcının yüklediği görüntü dosyalarını açmak ve işlemek için.
import numpy as np               # Görüntüleri sayısal dizilere (matris) çevirmek ve normalizasyon için.
import pandas as pd              # Tahmin olasılıklarını tablo yapısına dönüştürüp grafiğe hazırlamak için.
import plotly.express as px      # Tahmin sonuçlarını interaktif sütun grafiklerine dönüştürmek için.
import os                       # Dosya yolları, dizin kontrolü ve dosya varlık sorgulaması için.

# 1. SAYFA YAPILANDIRMASI (Madde 17 & 18: Estetik ve Gezinme)
# st.set_page_config: Tarayıcı sekmesindeki başlığı ve sayfanın geniş (wide) yerleşimini tanımlar.
st.set_page_config(page_title="Renal AI | Böbrek Analiz Pro", layout="wide")

# st.markdown(unsafe_allow_html=True): Sayfaya doğrudan CSS enjekte ederek medikal bir tema oluşturur.
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; } /* Uygulamanın ana arka plan rengini koyu lacivert yapar. */
    
    /* Üst taraftaki yatay navigasyon menüsünün (navbar) çerçeve ve arka plan stili. */
    div[data-testid="stHorizontalBlock"] {
        background-color: #161b22;
        padding: 10px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 30px;
    }
    
    /* Rapor ve bilgi bloklarını (medical-card) birbirinden ayıran kutu tasarımı. */
    .medical-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    /* Başlıkların (h1, h2, h3) medikal mavi tonlarında ve kalın görünmesini sağlar. */
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Inter', sans-serif; font-weight: 700; }
    
    /* Navigasyon ve analiz butonları için özel degrade (gradient) ve geçiş efektleri. */
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
    
    /* Fare butonun üzerine geldiğinde (hover) butonu hafifçe büyütür ve parlama verir. */
    .stButton>button:hover { transform: scale(1.01); box-shadow: 0 5px 15px rgba(88, 166, 255, 0.4); }
    </style>
    """, unsafe_allow_html=True)

# 2. MODEL YÜKLEME VE VERSİYON UYUMLULUK ÇÖZÜMÜ (Madde 19)
# @st.cache_resource: Modeli bir kez belleğe yükler; böylece her buton tıklamasında model tekrar yüklenip sistem donmaz.
@st.cache_resource 
def load_trained_model():
    model_path = 'kidney_disease_mobilenet_model2.h5' # Projenin ana dizinindeki model dosyasının ismi.
    
    # os.path.exists: Model dosyasının sunucuda fiziksel olarak bulunup bulunmadığını kontrol eder.
    if not os.path.exists(model_path): 
        st.error(f"❌ Model dosyası ({model_path}) bulunamadı!") # Dosya yoksa kullanıcıya uyarı verir.
        return None

    # --- KERAS 3 UYUMLULUK YAMASI (ZORUNLU TEKNİK OPERASYON) ---
    # TensorFlow'un yeni sürümlerinde (Keras 3), eski model parametreleri hata verebilir.
    from tensorflow.keras.layers import InputLayer # Modelin giriş katman sınıfını içe aktarır.

    # CompatibleInputLayer: Orijinal InputLayer'ı miras alarak (inheritance) hatalı parametreleri onarır.
    class CompatibleInputLayer(InputLayer):
        def __init__(self, *args, **kwargs): # Katman oluşturulurken gönderilen tüm argümanları yakalar.
            # 'batch_shape' anahtarı Keras 2'ye aittir; eğer varsa;
            if 'batch_shape' in kwargs:
                # .pop: 'batch_shape'i siler ve değerini Keras 3'ün tanıdığı 'batch_input_shape'e aktarır.
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
            # super().__init__: Onarılmış parametrelerle orijinal Keras katmanını başlatır.
            super().__init__(*args, **kwargs)

    try:
        # tf.keras.models.load_model: Kayıtlı h5 dosyasını derin öğrenme motoruna yükler.
        # compile=False: Modelin eğitim aşamasındaki (optimizer gibi) ayarlarını yüklemez, sadece tahmine odaklanır.
        # custom_objects: Sisteme "Hatalı InputLayer yerine benim düzelttiğim sınıfı kullan" talimatını verir.
        return tf.keras.models.load_model(
            model_path, 
            compile=False, 
            custom_objects={'InputLayer': CompatibleInputLayer}
        )
    except Exception as e:
        # Yükleme sırasında beklenmedik bir hata oluşursa bunu ekranda gösterir.
        st.error(f"⚠️ Model Yükleme Hatası: {e}")
        return None

# Fonksiyonu çağırarak modeli RAM'e (belleğe) alır.
model = load_trained_model()
# LABELS: Modelin tahmin ettiği sayısal indeksleri (0,1,2,3) insan dilindeki hastalıklara eşler.
LABELS = ['Cyst (Kist)', 'Normal', 'Stone (Taş)', 'Tumor (Tümör)']

# 3. ÜST NAVBAR (Yatay Navigasyon Menüsü)
st.title("🛡️ Renal AI: İleri Seviye Böbrek Analiz Sistemi")
# st.columns(4): Sayfanın üst kısmına butonlar için 4 eşit genişlikte sütun oluşturur.
menu_col1, menu_col2, menu_col3, menu_col4 = st.columns(4) 

# Butonlara tıklandığında 'st.session_state' (oturum durumu) güncellenerek sayfa içeriği değişir.
with menu_col1:
    if st.button("🏠 PROJE VİZYONU"): st.session_state.page = "vizyon" # Vizyon sayfasına yönlendirir.
with menu_col2:
    if st.button("🧠 TEKNİK ALTYAPI"): st.session_state.page = "teknik" # Teknik bilgilere yönlendirir.
with menu_col3:
    if st.button("📊 ANALİTİK RAPORLAR"): st.session_state.page = "analitik" # Grafiklere yönlendirir.
with menu_col4:
    if st.button("🔬 CANLI TANI MERKEZİ"): st.session_state.page = "tani" # Analiz modülüne yönlendirir.

# Eğer kullanıcı henüz bir seçim yapmadıysa varsayılan olarak "vizyon" sayfasını açar.
if 'page' not in st.session_state: st.session_state.page = "vizyon"
st.divider() # Navbar ile içerik arasına ince bir çizgi çeker.

# ==============================================================================
# 4. SAYFA İÇERİKLERİ VE MANTIK AKIŞI
# ==============================================================================

# --- BÖLÜM 1: VİZYON ---
if st.session_state.page == "vizyon":
    st.header("🏥 Bölüm 1: Problemin Tanımı ve Akademik Önemi")
    st.markdown('<div class="medical-card">', unsafe_allow_html=True) # Tasarlanan kart yapısını başlatır.
    c1, c2 = st.columns([2, 1]) # İçerik için biri geniş, biri dar iki sütun açar.
    with c1:
        st.subheader("📌 Problemin Tanımı")
        st.write("Böbrek patolojilerinin BT kesitlerinden teşhisi radyologlar için zaman alıcıdır. Renal AI, karar destek mekanizması olarak teşhis doğruluğunu artırır.")
    with c2:
        st.subheader("📚 Veri Seti")
        # st.info: Önemli bilgileri mavi bir kutu içinde gösterir.
        st.info("**Kaynak:** Kaggle CT-Kidney\n\n**Sınıf:** 4 (Normal, Kist, Taş, Tümör)")
    st.markdown('</div>', unsafe_allow_html=True) # Kart yapısını kapatır.

# --- BÖLÜM 2: TEKNİK ALTYAPI ---
elif st.session_state.page == "teknik":
    st.header("🧬 Bölüm 2: Mühendislik ve Model Altyapısı")
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    # Modelin teknik mimarisi hakkında özet bilgiler sunar.
    st.write("- **Model:** MobileNetV2 (Transfer Learning - Hazır eğitilmiş ağırlıklar kullanıldı).")
    st.write("- **Ön İşleme:** 1/255 Normalizasyon (Pikselleri 0-1 arasına çeker) & 224x224 Resize.")
    st.write("- **Optimizasyon:** Adam Optimizer (Öğrenme hızı: 0.0001 olarak ayarlandı).")
    st.markdown('</div>', unsafe_allow_html=True)

# --- BÖLÜM 3: ANALİTİK RAPORLAR (Metrikler ve Grafikler) ---
elif st.session_state.page == "analitik":
    st.header("📊 Bölüm 3: Model Başarım Metrikleri")
    m1, m2, m3, m4 = st.columns(4) # Sayısal başarı metrikleri için 4 kolon.
    m1.metric("Doğruluk (Acc)", "%68") # Toplam doğru tahmin oranı.
    m2.metric("AUC Skoru", "0.94")     # Modelin sınıfları birbirinden ayırma kabiliyeti.
    m3.metric("F1-Skoru", "0.65")      # Kesinlik ve duyarlılığın harmonik ortalaması.
    m4.metric("Duyarlılık", "0.88")    # Hastaları (pozitifleri) bulma başarısı.
    st.divider()
    
    # os.path.exists: Metrik kodundan gelen PNG dosyalarının varlığını kontrol eder.
    if os.path.exists('learning_curves.png'): 
        st.image('learning_curves.png', caption="Eğitim Grafikleri (Accuracy/Loss)", width=1100)
    
    g1, g2 = st.columns(2) # Karmaşıklık matrisi ve ROC eğrisi için yan yana iki alan.
    with g1:
        if os.path.exists('kidney_confusion_matrix.png'): 
            st.image('kidney_confusion_matrix.png', caption="Karmaşıklık Matrisi (Hata Analizi)", use_container_width=True)
    with g2:
        if os.path.exists('kidney_roc_curve.png'): 
            st.image('kidney_roc_curve.png', caption="ROC Analizi (Performans Eğrisi)", use_container_width=True)

# --- BÖLÜM 4: CANLI TANI MERKEZİ ---
elif st.session_state.page == "tani":
    st.header("🔬 Bölüm 4: BT Kesiti Canlı Analiz Modülü")
    st.markdown('<div class="medical-card">', unsafe_allow_html=True)
    # st.file_uploader: Kullanıcının yerel cihazından BT (CT) görüntüsünü sisteme yüklemesini sağlar.
    up_file = st.file_uploader("Analiz edilecek BT görüntüsünü yükleyiniz...", type=["jpg", "png", "jpeg"])
    
    if up_file: # Eğer dosya başarıyla yüklendiyse;
        col_img, col_analysis = st.columns([1, 1], gap="large") # Resim ve analizi yan yana koyar.
        img = Image.open(up_file).convert('RGB') # Yüklenen dosyayı açar ve RGB kanalına çevirir (standartlaştırma).
        
        with col_img:
            st.markdown("### 🖼️ Giriş BT Kesiti")
            st.image(img, caption="Yüklenen Ham Görüntü", width=420) # Genişliği sabitleyerek arayüz kaymasını engeller.
        
        with col_analysis:
            st.markdown("### 🤖 Analiz Sonuçları")
            if model is not None: # Model başarıyla belleğe yüklendiyse;
                if st.button("🚀 ANALİZİ BAŞLAT"): # Kullanıcı "Analiz Et" butonuna bastığında;
                    with st.spinner('Yapay Zeka Taraması Yapılıyor...'): # İşlem sırasında bekleme animasyonu gösterir.
                        
                        # --- GÖRÜNTÜ ÖN İŞLEME (PREPROCESSING) ---
                        # img.resize: Resmi modelin eğitildiği standart boyuta (224x224) getirir.
                        # np.array / 255.0: Piksel değerlerini 0-255 arasından 0-1 arasına (float32) çeker.
                        p_img = np.array(img.resize((224, 224))).astype('float32') / 255.0
                        
                        # np.expand_dims: Modelin beklediği (1, 224, 224, 3) şekline (batch boyutu) getirir.
                        p_img = np.expand_dims(p_img, axis=0) 
                        
                        # --- TAHMİN (INFERENCE) ---
                        # model.predict: Model, resimdeki patoloji olasılıklarını bir dizi olarak hesaplar.
                        preds = model.predict(p_img, verbose=0) 
                        
                        # np.argmax: Olasılık dizisindeki (Örn: [0.1, 0.8, 0.05, 0.05]) en yüksek değerin indeksini bulur.
                        idx = np.argmax(preds)                 
                        
                        # --- SONUÇ RAPORLAMA ---
                        # res_color: Sonuca göre dinamik renk seçer (Normal ise yeşil, değilse kırmızı).
                        res_color = "#238636" if idx == 1 else "#da3633"
                        st.markdown(f"""
                            <div style="border-left: 10px solid {res_color}; padding: 20px; background-color: #1f2937; border-radius: 12px; margin-top:0;">
                                <h2 style="margin:0; color: white !important;">Teşhis: {LABELS[idx]}</h2>
                                <h3 style="margin:0; color: #8b949e !important;">Güven Oranı: %{np.max(preds)*100:.1f}</h3>
                            </div>
                        """, unsafe_allow_html=True) # HTML kodunu ekrana güvenli bir şekilde basar.
                        
                        # --- GRAFİKSEL GÖSTERİM ---
                        # pd.DataFrame: Olasılıkları grafik kütüphanesinin anlayacağı tablo yapısına sokar.
                        df_p = pd.DataFrame({'Patoloji': LABELS, 'Olasılık': preds[0]})
                        
                        # px.bar: Sınıf olasılıklarını interaktif bir sütun grafiği olarak çizer.
                        fig = px.bar(df_p, x='Patoloji', y='Olasılık', color='Patoloji', height=300, template="plotly_dark")
                        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=10, b=10))
                        st.plotly_chart(fig, use_container_width=True) # Hazırlanan grafiği sayfaya yerleştirir.
            else:
                st.error("Model yüklenemedi. Lütfen sistem yöneticisiyle iletişime geçin.")
    st.markdown('</div>', unsafe_allow_html=True) # Kart yapısını kapatır.

# 5. ALT BİLGİ (Madde 20: Akademik Kimlik)
st.divider() # Sayfanın en altına ince bir çizgi çeker.
# HTML ile sayfa altına geliştirici bilgileri ve akademik künye eklenir.
st.markdown(f"<div style='text-align: center; color: #8b949e;'><b>Geliştirici:</b> Oğuzhan Dursun - 220706037 | <b>Ders:</b> Yapay Zeka ile Sağlık Bilişimi Vize Ödevi - 2026</div>", unsafe_allow_html=True)
