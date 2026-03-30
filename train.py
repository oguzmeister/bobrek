import os                                               # Dosya ve dizin işlemleri için kütüphane
import numpy as np                                      # Diziler ve sayısal hesaplamalar için kütüphane
import tensorflow as tf                                 # Derin öğrenme altyapısı
from tensorflow.keras import layers, models, callbacks  # Model katmanları ve eğitim araçları
from tensorflow.keras.applications import MobileNetV2   # Hazır eğitilmiş MobileNetV2 mimarisi
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Görüntü ön işleme araçları
from sklearn.utils import class_weight                  # Veri setindeki dengesizliği gidermek için araçlar
from sklearn.metrics import classification_report       # Başarı metrikleri raporlama aracı

# 1. PARAMETRELER VE VERİ YOLLARI
# Dataset'in bulunduğu klasör yolu tanımlanır
DATASET_PATH = r'C:\Users\Oguz\Desktop\archive (2)\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone'
IMG_SIZE = (224, 224)                                   # Görüntülerin yeniden boyutlandırılacağı standart ölçü
BATCH_SIZE = 32                                         # Her eğitim adımında işlenecek görüntü sayısı
EPOCHS = 25                                             # Tüm veri setinin eğitimde kaç kez taranacağı

# GÖRÜNTÜ ARTIRMA VE ÖN İŞLEME AYARLARI
# ImageDataGenerator ile veriler normalize edilir ve rastgele manipülasyonlar tanımlanır
train_datagen = ImageDataGenerator(
    rescale=1./255,                                     # Piksel değerleri 0-255 arasından 0-1 arasına çekilir
    rotation_range=40,                                  # Görüntüler rastgele 40 dereceye kadar döndürülür
    width_shift_range=0.2,                              # Yatayda rastgele kaydırma yapılır
    height_shift_range=0.2,                             # Dikeyde rastgele kaydırma yapılır
    shear_range=0.2,                                    # Kesme/bükme dönüşümü uygulanır
    zoom_range=0.3,                                     # Rastgele yakınlaştırma uygulanır
    horizontal_flip=True,                               # Yatayda rastgele aynalama yapılır
    vertical_flip=True,                                 # Dikeyde rastgele aynalama yapılır
    fill_mode='nearest',                                # Boş kalan pikseller en yakın değerle doldurulur
    validation_split=0.2                                # Verinin %20'si doğrulama için ayrılır
)

# EĞİTİM VERİSİ OLUŞTURUCU
# Tanımlanan ayarlara göre eğitim görüntüleri klasörden çekilir
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,                                       # Veri seti yolu
    target_size=IMG_SIZE,                               # Hedef boyut
    batch_size=BATCH_SIZE,                              # Yığın boyutu
    class_mode='categorical',                           # Çok sınıflı sınıflandırma türü
    subset='training',                                  # Eğitim alt kümesi olduğu belirtilir
    shuffle=True                                        # Veriler her seferinde karıştırılır
)

# DOĞRULAMA VERİSİ OLUŞTURUCU
# Modelin eğitim sırasında hiç görmediği verilerle test edilmesi için oluşturulur
val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,                                       # Veri seti yolu
    target_size=IMG_SIZE,                               # Hedef boyut
    batch_size=BATCH_SIZE,                              # Yığın boyutu
    class_mode='categorical',                           # Çok sınıflı sınıflandırma türü
    subset='validation',                                # Doğrulama alt kümesi olduğu belirtilir
    shuffle=False                                       # Tahmin raporu için sıra korunur
)

# SINIF AĞIRLIKLARININ HESAPLANMASI
# Veri setindeki az olan sınıfların model tarafından daha çok önemsenmesi sağlanır
weights = class_weight.compute_class_weight(
    class_weight='balanced',                            # Dengeli ağırlıklandırma modu
    classes=np.unique(train_generator.classes),         # Mevcut sınıfların listesi
    y=train_generator.classes                           # Verideki sınıf dağılımı
)
# Hesaplanan ağırlıklar Python sözlüğü (dictionary) formatına çevrilir
class_weights_dict = dict(enumerate(weights))
print(f"Sinif Agirliklari Hazir: {class_weights_dict}")

# 2. MOBILENET V2 MODEL MİMARİSİ
# ImageNet veri seti ile eğitilmiş MobileNetV2 temel model olarak yüklenir
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False                            # Temel modelin katmanları dondurulur

# Ardışık model yapısı tanımlanır
model = models.Sequential([
    base_model,                                         # Önceden eğitilmiş temel model eklenir
    layers.GlobalAveragePooling2D(),                    # 3 boyutlu matris 1 boyutlu vektöre indirgenir
    layers.BatchNormalization(),                        # Eğitim hızını artıran normalizasyon katmanı
    layers.Dense(256, activation='relu'),               # 256 nöronlu gizli katman
    layers.Dropout(0.5),                                # Aşırı öğrenmeyi engellemek için nöronların yarısı kapatılır
    layers.Dense(4, activation='softmax')               # 4 farklı hastalık sınıfı için çıkış katmanı
])

# Modelin derlenmesi
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Adam optimize edici ve öğrenme hızı ayarı
    loss='categorical_crossentropy',                    # Çok sınıflı kayıp fonksiyonu
    metrics=['accuracy']                                # Başarı takip metriği: Doğruluk
)

# 3. EĞİTİM SÜRECİ
# Doğrulama kaybı 7 devir boyunca düşmezse eğitimi durdurur ve en iyi ağırlıkları yükler
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

print("Egitim Basliyor...")
# Modelin eğitimi başlatılır
history = model.fit(
    train_generator,                                    # Eğitim verisi
    epochs=EPOCHS,                                      # Toplam devir sayısı
    validation_data=val_generator,                      # Doğrulama verisi
    class_weight=class_weights_dict,                    # Hesaplanan sınıf ağırlıkları uygulanır
    callbacks=[early_stop]                              # Erken durdurma kuralı eklenir
)

# 4. MODELİN KAYDEDİLMESİ
# Eğitilen model h5 formatında disk üzerine kaydedilir
model.save('kidney_disease_mobilenet_model2.h5')
print("Model basariyla kaydedildi.")

# 5. PERFORMANS RAPORLAMASI
val_generator.reset()                                   # Generator indeksi başa sarılır
Y_pred = model.predict(val_generator)                   # Modelin tahmin olasılıkları alınır
y_pred = np.argmax(Y_pred, axis=1)                      # En yüksek olasılıklı sınıf indeksi seçilir

# Sınıf bazlı detaylı performans raporu yazdırılır
print("Performans Raporu:")
print(classification_report(val_generator.classes, y_pred, target_names=list(val_generator.class_indices.keys())))