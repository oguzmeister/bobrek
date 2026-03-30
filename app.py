import streamlit as st            # Web tabanlı kullanıcı arayüzünü (UI) oluşturmak için temel kütüphane.
import tensorflow as tf           # Derin öğrenme modelini (Keras) arka planda çalıştırmak için.
from PIL import Image             # Kullanıcının yüklediği görüntü dosyalarını açmak ve işlemek için.
import numpy as np                # Görüntüleri sayısal dizilere (matris) çevirmek ve normalizasyon için.
import pandas as pd               # Tahmin olasılıklarını tablo yapısına dönüştürüp grafiğe hazırlamak için.
import plotly.express as px       # Tahmin sonuçlarını interaktif sütun grafiklerine dönüştürmek için.
import os                         # Dosya yolları, dizin kontrolü ve dosya varlık sorgulaması için.

# 1. SAYFA YAPILANDIRMASI (Madde 17 & 18: Estetik ve Gezinme)
st.set_page_config(page_title="Böbrek Analiz Pro", layout="wide") # Tarayıcı sekme başlığını ve sayfa yerleşimini geniş mod olarak ayarlar.

# --- GELİŞMİŞ TIBBİ PANEL CSS (BOŞLUKLAR DÜZELTİLDİ) ---
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; } 
    
    /* Üst menü (Navbar) için görsel stil düzenlemeleri */
    div[data-testid="stHorizontalBlock"] {
        background-color: #161b22;
        padding: 10px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 30px;
    }
    
    /* İçeriklerin içine yerleştiği tıbbi kart yapısı */
    .medical-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    /* Kart başlıklarının (h1, h2, h3) üst boşluklarını sıfırlayarak hizalamayı düzeltir */
    .medical-card h1, .medical-card h2, .medical-card h3 {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }
    
    /* Başlıkların rengini mavi tonlarına çeker ve yazı tipini belirler */
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Inter', sans-serif; font-weight: 700; }
    
    /* Streamlit butonlarının genel görünümünü modern ve degrade geçişli yapar */
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
    
    /* Fare butona geldiğinde hafif büyüme ve parlama efekti verir */
    .stButton>button:hover { transform: scale(1.01); box-shadow: 0 5px 15px rgba(88, 166, 255, 0.4); }
    </style>
    """, unsafe_allow_html=True) # CSS kodlarının Streamlit tarafından işlenmesini sağlar.

# 2. MODEL YÜKLEME VE VERSİYON UYUMLULUK ÇÖZÜMÜ (Madde 19)
@st.cache_resource # Modelin belleğe bir kez alınmasını sağlar, her sayfa değişiminde tekrar yükleme yapmaz.
def load_trained_model():
    model_path = 'kidney_disease_mobilenet_model2.h5' # Yüklenecek model dosyasının tam adı.
    if not os.path.exists(model_path): # Eğer belirtilen isimde bir dosya klasörde yoksa:
        st.error(f"❌ Model dosyası ({model_path}) bulunamadı!") # Kullanıcıya dosyanın eksik olduğunu bildirir.
        return None # Fonksiyonu sonlandırır.

    from tensorflow.keras.layers import InputLayer # Modelin giriş katmanı tipini içeri aktarır.
    class CompatibleInputLayer(InputLayer): # Yeni Keras sürümlerinde değişen parametre isimleri için köprü görevi görür.
        def __init__(self, *args, **kwargs): # Katman başlatıldığında:
            if 'batch_shape' in kwargs: # Eğer eski sürümden kalan 'batch_shape' varsa:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape') # Onu yeni sürümün beklediği isme dönüştürür.
            super().__init__(*args, **kwargs) # Standart katman özelliklerini korur.

    try:
        return tf.keras.models.load_model(
            model_path, 
            compile=False, 
            custom_objects={'InputLayer': CompatibleInputLayer}
        ) # Modeli derlemeden (inference için) ve uyumluluk katmanını kullanarak yükler.
    except Exception as e: # Eğer dosya bozuksa veya yüklenemiyorsa hata yakalar:
        st.error(f"⚠️ Model Yükleme Hatası: {e}") # Hata detayını ekrana yazdırır.
        return None # Boş değer döndürür.

model = load_trained_model() # Yukarıdaki fonksiyonu çağırarak modeli 'model' değişkenine atar.
LABELS = ['Cyst (Kist)', 'Normal', 'Stone (Taş)', 'Tumor (Tümör)'] # Modelin tahmin edeceği sınıfların listesi.

# 3. ÜST NAVBAR (Navigasyon Menüsü)
st.title("🛡️ Böbrek Analiz Pro: İleri Seviye Böbrek Analiz Sistemi") # Ana uygulama başlığı.
menu_col1, menu_col2, menu_col3, menu_col4 = st.columns(4) # Menü butonlarını yan yana dizmek için 4 sütun oluşturur.

with menu_col1:
    if st.button("🏠 PROJE VİZYONU"): st.session_state.page = "vizyon" # Butona tıklandığında sayfa durumunu 'vizyon' yapar.
with menu_col2:
    if st.button("🧠 TEKNİK ALTYAPI"): st.session_state.page = "teknik" # Butona tıklandığında sayfa durumunu 'teknik' yapar.
with menu_col3:
    if st.button("📊 ANALİTİK RAPORLAR"): st.session_state.page = "analitik" # Butona tıklandığında sayfa durumunu 'analitik' yapar.
with menu_col4:
    if st.button("🔬 CANLI TANI MERKEZİ"): st.session_state.page = "tani" # Butona tıklandığında sayfa durumunu 'tani' yapar.

if 'page' not in st.session_state: st.session_state.page = "vizyon" # Uygulama ilk açıldığında varsayılan olarak vizyon sayfasını seçer.
st.divider() # Menü ile içerik arasına yatay çizgi çeker.

# ==============================================================================
# 4. SAYFA İÇERİKLERİ
# ==============================================================================

# --- BÖLÜM 1: VİZYON ---
if st.session_state.page == "vizyon": # Aktif sayfa vizyon ise bu blok çalışır.
    st.header("🏥 Bölüm 1: Problemin Tanımı ve Akademik Önemi") # Bölüm başlığı.
    st.markdown('<div class="medical-card">', unsafe_allow_html=True) # İçeriği kart içine alır.
    c1, c2 = st.columns([2, 1]) # İçeriği %66 ve %33 genişliğinde iki sütuna böler.
    with c1:
        st.subheader("📌 Problemin Tanımı") # Alt başlık.
        st.write("""
        Günümüzde böbrek hastalıklarının teşhisinde **Bilgisayarlı Tomografi (BT)** kesitlerinin manuel incelenmesi, radyologlar üzerinde ciddi bir bilişsel yük oluşturmaktadır. 
        Her bir hastaya ait yüzlerce kesit görüntüsünün titizlikle taranması; yorgunluk, dikkat dağınıklığı ve sınırlı zaman gibi faktörlere bağlı olarak **tıbbi hata (diagnostic error)** riskini beraberinde getirmektedir. 
        Özellikle erken evre kist, taş ve tümör oluşumlarının birbirine benzer görsel doku özellikleri göstermesi, teşhis sürecini karmaşıklaştırmaktadır.
        
        **Böbrek Analiz Pro**, Sağlık Bilişimi ve Derin Öğrenme prensiplerini kullanarak bu soruna dijital bir çözüm sunar. Sistem, BT görüntülerini piksel düzeyinde analiz ederek patolojik anomali potansiyeli taşıyan bölgeleri saniyeler içinde sınıflandırır. 
        Bu çalışma, bir doktorun yerini almaktan ziyade, hekimlerin karar verme süreçlerini hızlandıran ve teşhis doğruluğunu valide eden bir **Klinik Karar Destek Mekanizması** olarak tasarlanmıştır. Mimari olarak, hafif yapısı ve yüksek işlem hızı nedeniyle mobil/web tabanlı çalışmaya en uygun seçenek olan MobileNetV2 tercih edilmiştir. Transfer Learning yaklaşımıyla kurulan bu mimaride, temel modelin üzerine GlobalAveragePooling2D, BatchNormalization ve Dropout katmanları eklenerek aşırı öğrenmenin önüne geçilmiştir. Eğitim sürecinde Adam optimizasyon algoritması ve $0.0001$ öğrenme hızı (learning rate) tercih edilmiş; $25$ epoch ve $32$ batch size parametreleriyle modelin stabil bir şekilde öğrenmesi sağlanmıştır.
        """) # Metin içeriğini yazar.
    with c2:
        st.subheader("📚 Veri Seti") # Veri kaynağı başlığı.
        st.info("""
                Görüntü Türü: Veriler, bilgisayarlı tomografi (BT/CT) cihazlarından elde edilen dikine kesit görüntülerinden oluşmaktadır.

                Toplam Örnek Sayısı: Veri seti içerisinde toplamda 12.446 adet benzersiz BT görüntüsü bulunmaktadır.

                Sınıf Sayısı: Modelin eğitilmesi ve test edilmesi sürecinde 4 farklı kategorik sınıf kullanılmıştır.
                """) # Veri seti linkini ve bilgilerini mavi bir bilgi kutusunda sunar.

    # --- KAYNAKÇA (Madde 20) ---
    st.divider() # Bölüm sonu ayırıcı.
    st.subheader("📚 Kaynakça ve Literatür Taraması") # Akademik kaynaklar başlığı.
    st.write("""
    1.  **Sandler, M., et al. (2018).** *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* [Makale Erişimi](https://arxiv.org/abs/1801.04381)
    2.  **Kaggle Dataset:** *CT Kidney Dataset: Normal-Cyst-Tumor and Stone.* [Veri Seti Erişimi](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone)
    """) # Kullanılan kaynakların listesini yazar.
    st.markdown('</div>', unsafe_allow_html=True) # Kart yapısını kapatır.

# --- BÖLÜM 2: TEKNİK ALTYAPI ---
elif st.session_state.page == "teknik": # Aktif sayfa teknik ise bu blok çalışır.
    st.header("🧬 Bölüm 2: Derin Öğrenme Metodolojisi ve Sistem Mimarisi") # Bölüm başlığı.
    st.markdown('<div class="medical-card">', unsafe_allow_html=True) # Kart yapısı başlatılır.
    col_arch, col_prep = st.columns(2) # Eşit genişlikte iki sütun oluşturur.
    with col_arch:
        st.subheader("🧠 Model Mimarisi: MobileNetV2") # Model yapısı başlığı.
        st.write("""
        Bu projede, Google tarafından geliştirilen ve **Transfer Learning** prensibiyle çalışan **MobileNetV2** mimarisi tercih edilmiştir. 
        MobileNetV2, 'Inverted Residuals' yapısı sayesinde düşük hesaplama gücüyle yüksek doğruluk oranlarına ulaşabilen optimize bir CNN mimarisidir.
        Veri Ayrımı: Modelin başarısını objektif bir şekilde ölçmek amacıyla toplam verinin %80'i eğitim (training), kalan %20'si ise doğrulama (validation) seti olarak ayrılmıştır.

        Yeniden Boyutlandırma: Orijinal BT görüntüleri, MobileNetV2 mimarisinin standart giriş boyutu olan 224x224 piksel boyutuna sabitlenmiştir.
        
        Normalizasyon: Piksel yoğunluk değerleri (0-255), hesaplama hızını ve model kararlılığını artırmak için 1/255 katsayısı ile çarpılarak 0-1 aralığına çekilmiştir.
        
        Veri Artırma (Augmentation): Eğitim setindeki çeşitliliği artırmak ve aşırı öğrenmeyi (overfitting) engellemek için resimlere rastgele döndürme (rotation), yakınlaştırma (zoom) ve yatay/dikey aynalama (flip) işlemleri uygulanmıştır.
        """) # Mimari seçim gerekçesini açıklar.
    with col_prep:
        st.subheader("⚙️ Veri Ön İşleme ve Optimizasyon") # Hazırlık süreci başlığı.
        st.write("""
        **1. Geometrik Düzenleme:** Görüntüler **224x224** boyutuna sabitlenmiştir.
        **2. Normalizasyon:** Piksel değerleri **1/255** katsayısıyla **[0, 1]** aralığına çekilmiştir.
        **3. Adam Optimizer:** Öğrenme hızı **0.0001** olarak set edilmiştir.
        """) # Teknik ön işleme adımlarını yazar.
    
    # --- EĞİTİM DETAYLARI (Madde 7, 10, 11) ---
    st.divider() # Ayırıcı.
    st.subheader("📊 Eğitim Parametreleri ve Veri Ayrımı") # Eğitim metrikleri başlığı.
    st.write("""
    - **Veri Ayrımı:** %80 Eğitim, %20 Doğrulama (Validation).
    - **Hiperparametreler:** Batch Size: 32, Epoch: 25, Loss: Categorical Cross-Entropy.
    - **Eğitim Stratejisi:** 'Early Stopping' kullanılarak overfitting engellenmiş ve en iyi ağırlıklar kaydedilmiştir.
    """) # Eğitimin nasıl yapıldığını listeler.
    st.markdown('</div>', unsafe_allow_html=True) # Kart yapısını kapatır.

# --- BÖLÜM 3: ANALİTİK RAPORLAR ---
# --- BÖLÜM 3: ANALİTİK RAPORLAR ---
elif st.session_state.page == "analitik": # Kullanıcı navigasyondan 'Analitik Raporlar' sayfasını seçtiyse bu blok çalışır.
    st.header("📊 Bölüm 3: Model Başarım Metrikleri ve Grafik Yorumları") # Sayfa ana başlığını oluşturur.
    
    # Madde 12 & 13: Performans Metriklerinin Sunumu
    m1, m2, m3, m4 = st.columns(4) # Metrik özetlerini yan yana göstermek için 4 sütun oluşturur.
    m1.metric("Doğruluk (Acc)", "%68", help="Modelin tüm sınıflardaki genel doğru tahmin oranı.") # Genel doğruluk oranını gösterir.
    m2.metric("AUC Skoru", "0.94", help="Modelin sınıfları birbirinden ayırt etme gücü (1.0 mükemmeldir).") # AUC değerini gösterir.
    m3.metric("F1-Skoru", "0.65", help="Hassasiyet ve duyarlılık değerlerinin dengeli ortalaması.") # F1 skorunu gösterir.
    m4.metric("Duyarlılık", "0.88", help="Hastalık vakalarını yakalama (atlamama) yeteneği.") # Duyarlılık (Recall) oranını gösterir.
    
    st.divider() # Metrikler ile grafikler arasına yatay ayırıcı çizgi ekler.

    # Madde 14 & 15: Eğitim Grafikleri ve Teknik Yorumlanması
    if os.path.exists('learning_curves.png'): # Eğitim sürecini gösteren grafik dosyası dizinde var mı kontrol eder.
        st.subheader("📈 Eğitim Süreci Analizi") # Grafik bölümü için alt başlık oluşturur.
        st.image('learning_curves.png', caption="Şekil 1: Eğitim ve Doğrulama (Accuracy/Loss) Grafikleri", width=1100) # Kayıtlı grafiği ekrana basar.
        st.info("""
            **Teknik Yorum:** Accuracy grafiğinde eğitim ve doğrulama eğrilerinin birbirine yakın seyretmesi, modelin genelleme yeteneğinin yüksek olduğunu ve 'aşırı öğrenme' (overfitting) sorununun minimize edildiğini gösterir.
            """) # Grafiğin akademik ve teknik yorumunu bilgi kutusunda sunar.

    # Madde 16: Grafik Kalitesi ve Düzeni (Matris ve ROC yan yana)
    g1, g2 = st.columns(2) # Karmaşıklık matrisi ve ROC eğrisi için yan yana iki sütun oluşturur.
    
    with g1: # Sol sütun işlemleri.
        if os.path.exists('kidney_confusion_matrix.png'): # Karmaşıklık matrisi dosyası var mı kontrol eder.
            st.subheader("🎯 Hata Matrisi Analizi") # Matris için alt başlık oluşturur.
            st.image('kidney_confusion_matrix.png', use_container_width=True) # Matris görselini sütun genişliğine göre yerleştirir.
            st.write("""
                **Teknik Yorum:** Karmaşıklık matrisi incelendiğinde, modelin özellikle 'Normal' kesitleri yüksek doğrulukla ayırt ettiği; ancak 'Cyst' ve 'Tumor' gibi benzer doku özelliklerine sahip sınıflar arasında sınırlı bir karışıklık yaşadığı gözlemlenmiştir.
                """) # Matris sonuçlarını teknik açıdan yorumlar.

    with g2: # Sağ sütun işlemleri.
        if os.path.exists('kidney_roc_curve.png'): # ROC eğrisi dosyası var mı kontrol eder.
            st.subheader("📉 ROC Eğrisi Analizi") # ROC eğrisi için alt başlık oluşturur.
            st.image('kidney_roc_curve.png', use_container_width=True) # ROC görselini sütun genişliğine göre yerleştirir.
            st.write("""
                **Teknik Yorum:** 0.94'lük AUC değeri, modelin her bir patolojiyi rastgele tahminden çok daha üstün bir başarıyla (mükemmele yakın) sınıflandırabildiğini ve ayırt etme eşiğinin yüksek olduğunu kanıtlamaktadır.
                """) # ROC analizini akademik bütünlükle yorumlar.

# --- BÖLÜM 4: CANLI TANI MERKEZİ ---
elif st.session_state.page == "tani": # Aktif sayfa tanı merkezi ise bu blok çalışır.
    st.header("🔬 Bölüm 4: BT Kesiti Canlı Analiz Modülü") # Bölüm başlığı.
    st.markdown('<div class="medical-card">', unsafe_allow_html=True) # Kart yapısı.
    up_file = st.file_uploader("Analiz edilecek BT görüntüsünü yükleyiniz...", type=["jpg", "png", "jpeg"]) # Bilgisayardan resim seçme alanı.
    if up_file: # Kullanıcı bir dosya seçtiyse:
        col_img, col_analysis = st.columns([1, 1], gap="large") # Görüntü ve sonucu yan yana koymak için iki sütun.
        img = Image.open(up_file).convert('RGB') # Yüklenen resmi açar ve RGB formatına çevirir.
        with col_img:
            st.subheader("🖼️ Giriş BT Kesiti") # Resim başlığı.
            st.image(img, caption="Yüklenen Ham Görüntü", width=420) # Kullanıcının yüklediği resmi ekrana basar.
        with col_analysis:
            st.subheader("🤖 Analiz Sonuçları") # Sonuç başlığı.
            if model is not None: # Eğer model dosyası başarıyla okunabildiyse:
                if st.button("🚀 ANALİZİ BAŞLAT"): # Analiz butonuna basıldığında:
                    with st.spinner('Yapay Zeka Taraması Yapılıyor...'): # İşlem bitene kadar dönen bir yükleme simgesi çıkarır.
                        p_img = np.array(img.resize((224, 224))).astype('float32') / 255.0 # Resmi modele uygun boyuta getirip normalize eder.
                        p_img = np.expand_dims(p_img, axis=0) # Tek bir resim olduğu için boyutunu (1, 224, 224, 3) yapar.
                        preds = model.predict(p_img, verbose=0) # Modelden olasılık tahminlerini alır.
                        idx = np.argmax(preds) # En yüksek olasılığa sahip sınıfın numarasını (indeks) seçer.
                        res_color = "#238636" if idx == 1 else "#da3633" # Normal (1) ise yeşil, hastalık varsa kırmızı renk kodu belirler.
                        st.markdown(f"""
                            <div style="border-left: 10px solid {res_color}; padding: 20px; background-color: #1f2937; border-radius: 12px; margin-top:0;">
                                <h2 style="margin:0; color: white !important;">Teşhis: {LABELS[idx]}</h2>
                                <h3 style="margin:0; color: #8b949e !important;">Güven Oranı: %{np.max(preds)*100:.1f}</h3>
                            </div>
                        """, unsafe_allow_html=True) # Teşhisi ve güven oranını şık bir kutu içinde gösterir.
                        df_p = pd.DataFrame({'Patoloji': LABELS, 'Olasılık': preds[0]}) # Olasılıkları grafiğe dökmek için tablo yapar.
                        fig = px.bar(df_p, x='Patoloji', y='Olasılık', color='Patoloji', height=300, template="plotly_dark") # Bar grafiği hazırlar.
                        fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=10, b=10)) # Grafiğin kenar boşluklarını ayarlar.
                        st.plotly_chart(fig, use_container_width=True) # Hazırlanan interaktif grafiği sayfaya ekler.
    st.markdown('</div>', unsafe_allow_html=True) # Kart yapısını kapatır.

# 5. ALT BİLGİ (Footer)
st.divider() # Sayfa sonu çizgisi.
st.markdown(f"<div style='text-align: center; color: #8b949e;'><b>Geliştirici:</b> Oğuzhan Dursun - 220706037 | <b>Ders:</b> Yapay Zeka ile Sağlık Bilişimi Vize Ödevi - 2026</div>", unsafe_allow_html=True) # Geliştirici ve ders bilgisini merkeze yazar.
