import streamlit as st            # Web tabanlı kullanıcı arayüzünü (UI) oluşturmak için temel kütüphane.
import tensorflow as tf           # Derin öğrenme modelini (Keras) arka planda çalıştırmak için.
from PIL import Image             # Kullanıcının yüklediği görüntü dosyalarını açmak ve işlemek için.
import numpy as np                # Görüntüleri sayısal dizilere (matris) çevirmek ve normalizasyon için.
import pandas as pd               # Tahmin olasılıklarını tablo yapısına dönüştürüp grafiğe hazırlamak için.
import plotly.express as px       # Tahmin sonuçlarını interaktif sütun grafiklerine dönüştürmek için.
import os                         # Dosya yolları, dizin kontrolü ve dosya varlık sorgulaması için.

# 1. SAYFA YAPILANDIRMASI (Madde 17 & 18: Estetik ve Gezinme)
st.set_page_config(page_title="Böbrek Analiz Pro", layout="wide") # Sayfa başlığını ayarlar ve geniş ekran modunu etkinleştirir.

# --- GELİŞMİŞ TIBBİ PANEL CSS (BOŞLUKLAR DÜZELTİLDİ) ---
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; } 
    
    /* Navbar tasarımı için CSS kuralları */
    div[data-testid="stHorizontalBlock"] {
        background-color: #161b22;
        padding: 10px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 30px;
    }
    
    /* Kart yapısı ve iç boşluk düzenlemesi için CSS kuralları */
    .medical-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    /* Kart içindeki başlıkların üstündeki boş bar etkisini kaldırmak için margin sıfırlama */
    .medical-card h1, .medical-card h2, .medical-card h3 {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }
    
    /* Başlık renkleri ve yazı tipi ayarları */
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Inter', sans-serif; font-weight: 700; }
    
    /* Buton tasarımı ve renk geçişleri */
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
    
    /* Buton üzerine gelindiğinde oluşan efektler */
    .stButton>button:hover { transform: scale(1.01); box-shadow: 0 5px 15px rgba(88, 166, 255, 0.4); }
    </style>
    """, unsafe_allow_html=True) # HTML ve CSS kodlarını Streamlit uygulamasına enjekte eder.

# 2. MODEL YÜKLEME VE VERSİYON UYUMLULUK ÇÖZÜMÜ (Madde 19)
@st.cache_resource # Modelin her seferinde yeniden yüklenmesini önlemek için önbelleğe alır.
def load_trained_model():
    model_path = 'kidney_disease_mobilenet_model2.h5' # Yüklenecek model dosyasının adı.
    if not os.path.exists(model_path): # Model dosyasının belirtilen yolda olup olmadığını kontrol eder.
        st.error(f"❌ Model dosyası ({model_path}) bulunamadı!") # Dosya yoksa hata mesajı gösterir.
        return None # Fonksiyondan boş değer döner.

    from tensorflow.keras.layers import InputLayer # Keras giriş katmanı sınıfını içe aktarır.
    class CompatibleInputLayer(InputLayer): # Yeni Keras versiyonlarıyla uyumluluk için özel bir katman sınıfı tanımlar.
        def __init__(self, *args, **kwargs): # Yapıcı metod.
            if 'batch_shape' in kwargs: # Eğer 'batch_shape' anahtarı varsa:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape') # Eski sürüm ismini yeni sürüm ismiyle değiştirir.
            super().__init__(*args, **kwargs) # Üst sınıfın yapıcı metodunu çağırır.

    try:
        return tf.keras.models.load_model(
            model_path, 
            compile=False, 
            custom_objects={'InputLayer': CompatibleInputLayer}
        ) # Modeli, özel uyumluluk katmanını kullanarak yükler ve döndürür.
    except Exception as e: # Yükleme sırasında bir hata oluşursa:
        st.error(f"⚠️ Model Yükleme Hatası: {e}") # Hata mesajını ekrana basar.
        return None # Fonksiyondan boş değer döner.

model = load_trained_model() # Modeli yükleme fonksiyonunu çalıştırır ve 'model' değişkenine atar.
LABELS = ['Cyst (Kist)', 'Normal', 'Stone (Taş)', 'Tumor (Tümör)'] # Tahmin sınıflarının isimlerini içeren liste.

# 3. ÜST NAVBAR
st.title("🛡️ Böbrek Analiz Pro: İleri Seviye Böbrek Analiz Sistemi") # Uygulama ana başlığını oluşturur.
menu_col1, menu_col2, menu_col3, menu_col4 = st.columns(4) # Menü butonları için yan yana 4 sütun oluşturur.

with menu_col1:
    if st.button("🏠 PROJE VİZYONU"): st.session_state.page = "vizyon" # Vizyon sayfasına geçiş butonunu oluşturur.
with menu_col2:
    if st.button("🧠 TEKNİK ALTYAPI"): st.session_state.page = "teknik" # Teknik altyapı sayfasına geçiş butonunu oluşturur.
with menu_col3:
    if st.button("📊 ANALİTİK RAPORLAR"): st.session_state.page = "analitik" # Analitik raporlar sayfasına geçiş butonunu oluşturur.
with menu_col4:
    if st.button("🔬 CANLI TANI MERKEZİ"): st.session_state.page = "tani" # Canlı tanı sayfasına geçiş butonunu oluşturur.

if 'page' not in st.session_state: st.session_state.page = "vizyon" # Eğer bir sayfa seçilmemişse varsayılan olarak vizyonu atar.
st.divider() # Yatay bir ayırıcı çizgi ekler.

# ==============================================================================
# 4. SAYFA İÇERİKLERİ
# ==============================================================================

# --- BÖLÜM 1: VİZYON ---
if st.session_state.page == "vizyon": # Eğer aktif sayfa 'vizyon' ise:
    st.header("🏥 Bölüm 1: Problemin Tanımı ve Akademik Önemi") # Bölüm başlığını yazar.
    st.markdown('<div class="medical-card">', unsafe_allow_html=True) # Kart tasarımını başlatır.
    c1, c2 = st.columns([2, 1]) # İçeriği yerleştirmek için farklı genişlikte iki sütun açar.
    with c1:
        st.subheader("📌 Problemin Tanımı") # Alt başlık oluşturur.
        st.write("""
        Günümüzde böbrek hastalıklarının teşhisinde **Bilgisayarlı Tomografi (BT)** kesitlerinin manuel incelenmesi, radyologlar üzerinde ciddi bir bilişsel yük oluşturmaktadır. 
        Her bir hastaya ait yüzlerce kesit görüntüsünün titizlikle taranması; yorgunluk, dikkat dağınıklığı ve sınırlı zaman gibi faktörlere bağlı olarak **tıbbi hata (diagnostic error)** riskini beraberinde getirmektedir. 
        Özellikle erken evre kist, taş ve tümör oluşumlarının birbirine benzer görsel doku özellikleri göstermesi, teşhis sürecini karmaşıklaştırmaktadır.
        
        **Böbrek Analiz Pro**, Sağlık Bilişimi ve Derin Öğrenme prensiplerini kullanarak bu soruna dijital bir çözüm sunar. Sistem, BT görüntülerini piksel düzeyinde analiz ederek patolojik anomali potansiyeli taşıyan bölgeleri saniyeler içinde sınıflandırır. 
        Bu çalışma, bir doktorun yerini almaktan ziyade, hekimlerin karar verme süreçlerini hızlandıran ve teşhis doğruluğunu valide eden bir **Klinik Karar Destek Mekanizması** olarak tasarlanmıştır.
        """) # Problemi tanımlayan metni yazar.
    with c2:
        st.subheader("📚 Veri Seti") # Veri seti bilgisi için alt başlık.
        st.info("""
                **Kaynak:** [Kaggle CT-Kidney](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)
                
                **Sınıf:** 4 (Normal, Kist, Taş, Tümör)
                """) # Veri seti kaynağını ve sınıflarını bir bilgi kutusunda gösterir.

    # --- KAYNAKÇA (Madde 20) ---
    st.divider() # Yatay çizgi ekler.
    st.subheader("📚 Kaynakça ve Literatür Taraması") # Kaynakça başlığı.
    st.write("""
    1.  **Sandler, M., et al. (2018).** *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* [Makale Erişimi](https://arxiv.org/abs/1801.04381)
    2.  **Kaggle Dataset:** *CT Kidney Dataset: Normal-Cyst-Tumor and Stone.* [Veri Seti Erişimi](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-stone)
    """) # Literatür bilgilerini yazar.
    st.markdown('</div>', unsafe_allow_html=True) # Kart tasarımını kapatır.

# --- BÖLÜM 2: TEKNİK ALTYAPI ---
elif st.session_state.page == "teknik": # Eğer aktif sayfa 'teknik' ise:
    st.header("🧬 Bölüm 2: Derin Öğrenme Metodolojisi ve Sistem Mimarisi") # Bölüm başlığını yazar.
    st.markdown('<div class="medical-card">', unsafe_allow_html=True) # Kart tasarımını başlatır.
    col_arch, col_prep = st.columns(2) # Eşit genişlikte iki sütun oluşturur.
    with col_arch:
        st.subheader("🧠 Model Mimarisi: MobileNetV2") # Model mimarisi alt başlığı.
        st.write("""
        Bu projede, Google tarafından geliştirilen ve **Transfer Learning** prensibiyle çalışan **MobileNetV2** mimarisi tercih edilmiştir. 
        MobileNetV2, 'Inverted Residuals' yapısı sayesinde düşük hesaplama gücüyle yüksek doğruluk oranlarına ulaşabilen optimize bir CNN mimarisidir. 
        """) # Mimari hakkında bilgi verir.
    with col_prep:
        st.subheader("⚙️ Veri Ön İşleme ve Optimizasyon") # Ön işleme alt başlığı.
        st.write("""
        **1. Geometrik Düzenleme:** Görüntüler **224x224** boyutuna sabitlenmiştir.
        **2. Normalizasyon:** Piksel değerleri **1/255** katsayısıyla **[0, 1]** aralığına çekilmiştir.
        **3. Adam Optimizer:** Öğrenme hızı **0.0001** olarak set edilmiştir.
        """) # Veri hazırlık adımlarını yazar.
    
    # --- EĞİTİM DETAYLARI (Madde 7, 10, 11) ---
    st.divider() # Yatay çizgi ekler.
    st.subheader("📊 Eğitim Parametreleri ve Veri Ayrımı") # Eğitim detayları başlığı.
    st.write("""
    - **Veri Ayrımı:** %80 Eğitim, %20 Doğrulama (Validation).
    - **Hiperparametreler:** Batch Size: 32, Epoch: 25, Loss: Categorical Cross-Entropy.
    - **Eğitim Stratejisi:** 'Early Stopping' kullanılarak overfitting engellenmiş ve en iyi ağırlıklar kaydedilmiştir.
    """) # Eğitim parametrelerini listeler.
    st.markdown('</div>', unsafe_allow_html=True) # Kart tasarımını kapatır.

# --- BÖLÜM 3: ANALİTİK RAPORLAR ---
elif st.session_state.page == "analitik": # Eğer aktif sayfa 'analitik' ise:
    st.header("📊 Bölüm 3: Model Başarım Metrikleri") # Bölüm başlığını yazar.
    m1, m2, m3, m4 = st.columns(4) # 4 adet metrik kutusu için sütun oluşturur.
    m1.metric("Doğruluk (Acc)", "%68") # Doğruluk metriğini gösterir.
    m2.metric("AUC Skoru", "0.94") # AUC skorunu gösterir.
    m3.metric("F1-Skoru", "0.65") # F1 skorunu gösterir.
    m4.metric("Duyarlılık", "0.88") # Duyarlılık (Recall) metriğini gösterir.
    st.divider() # Yatay çizgi ekler.
    if os.path.exists('learning_curves.png'): st.image('learning_curves.png', caption="Eğitim Grafikleri", width=1100) # Eğitim grafiği varsa yükler.
    g1, g2 = st.columns(2) # Grafikler için iki sütun açar.
    with g1:
        if os.path.exists('kidney_confusion_matrix.png'): st.image('kidney_confusion_matrix.png', use_container_width=True) # Karmaşıklık matrisini yükler.
    with g2:
        if os.path.exists('kidney_roc_curve.png'): st.image('kidney_roc_curve.png', use_container_width=True) # ROC eğrisini yükler.

# --- BÖLÜM 4: CANLI TANI MERKEZİ ---
elif st.session_state.page == "tani": # Eğer aktif sayfa 'tani' ise:
    st.header("🔬 Bölüm 4: BT Kesiti Canlı Analiz Modülü") # Bölüm başlığını yazar.
    st.markdown('<div class="medical-card">', unsafe_allow_html=True) # Kart tasarımını başlatır.
    up_file = st.file_uploader("Analiz edilecek BT görüntüsünü yükleyiniz...", type=["jpg", "png", "jpeg"]) # Dosya yükleme widget'ını oluşturur.
    if up_file: # Eğer bir dosya yüklendiyse:
        col_img, col_analysis = st.columns([1, 1], gap="large") # Görüntü ve analiz sonuçları için iki sütun açar.
        img = Image.open(up_file).convert('RGB') # Yüklenen dosyayı RGB formatında açar.
        with col_img:
            st.subheader("🖼️ Giriş BT Kesiti") # Görsel başlığı.
            st.image(img, caption="Yüklenen Ham Görüntü", width=420) # Yüklenen resmi ekranda gösterir.
        with col_analysis:
            st.subheader("🤖 Analiz Sonuçları") # Analiz başlığı.
            if model is not None: # Eğer model başarıyla yüklendiyse:
                if st.button("🚀 ANALİZİ BAŞLAT"): # Analizi tetikleyen buton.
                    with st.spinner('Yapay Zeka Taraması Yapılıyor...'): # İşlem sırasında bekleme animasyonu gösterir.
                        p_img = np.array(img.resize((224, 224))).astype('float32') / 255.0 # Resmi yeniden boyutlandırır ve normalize eder.
                        p_img = np.expand_dims(p_img, axis=0) # Model girişi için resmi (1, 224, 224, 3) boyutuna getirir.
                        preds = model.predict(p_img, verbose=0) # Modelden tahmin alır.
                        idx = np.argmax(preds) # En yüksek olasılığa sahip sınıfın indeksini bulur.
                        res_color = "#238636" if idx == 1 else "#da3633" # Sonuç Normal (1) ise yeşil, değilse kırmızı renk belirler.
                        st.markdown(f"""
                            <div style="border-left: 10px solid {res_color}; padding: 20px; background-color: #1f2937; border-radius: 12px; margin-top:0;">
                                <h2 style="margin:0; color: white !important;">Teşhis: {LABELS[idx]}</h2>
                                <h3 style="margin:0; color: #8b949e !important;">Güven Oranı: %{np.max(preds)*100:.1f}</h3>
                            </div>
                        """, unsafe_allow_html=True) # Tahmin sonucunu ve güven oranını renkli bir kutuda gösterir.
                        df_p = pd.DataFrame({'Patoloji': LABELS, 'Olasılık': preds[0]}) # Olasılıkları tabloya dönüştürür.
                        fig = px.bar(df_p, x='Patoloji', y='Olasılık', color='Patoloji', height=300, template="plotly_dark") # Sütun grafiği oluşturur.
                        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=10, b=10)) # Grafik yerleşimini düzenler.
                        st.plotly_chart(fig, use_container_width=True) # Grafiği ekrana yansıtır.
    st.markdown('</div>', unsafe_allow_html=True) # Kart tasarımını kapatır.

# 5. ALT BİLGİ
st.divider() # Yatay ayırıcı çizgi ekler.
st.markdown(f"<div style='text-align: center; color: #8b949e;'><b>Geliştirici:</b> Oğuzhan Dursun - 220706037 | <b>Ders:</b> Yapay Zeka ile Sağlık Bilişimi Vize Ödevi - 2026</div>", unsafe_allow_html=True) # Sayfa altına imza ve ders bilgisini ortalı olarak yazar.
