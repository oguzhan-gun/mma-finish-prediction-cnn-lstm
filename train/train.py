import numpy as np
import cv2
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization,LSTM, TimeDistributed, Reshape, GlobalAveragePooling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.layers import GlobalAveragePooling2D, TimeDistributed

# Bitirişin olduğu ve olmadığı videoların bulunduğu dizinler
finished = "database/mma_finished_videos"
unfinished = "database/mma_unfinished"
# Sabit frame sayısı ve görüntü boyutu
MAX_FRAMES = 32
FRAME_SIZE = (128, 128)

# Video yükleme ve işleme fonksiyonu

def load_video_frames(video_path, max_frames=MAX_FRAMES, frame_size=FRAME_SIZE):
    
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return np.zeros((max_frames, frame_size[0], frame_size[1], 1), dtype=np.float32)
    
    # Videonun tamamından eşit aralıklarla frame seç
    indices = np.linspace(0, total_frames - 1, max_frames).astype(int)
    
    frames = []
    
    for idx in indices:
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            frame = np.zeros((frame_size[0], frame_size[1]), dtype=np.float32)
        else:
            frame = cv2.resize(frame, frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.astype(np.float32) / 255.0
        
        frame = np.expand_dims(frame, axis=-1)
        frames.append(frame)
    
    cap.release()
    
    return np.array(frames, dtype=np.float32)


# Video dosyalarını ve etiketlerini yükleme fonksiyonu
def load_data():
    X_data = []
    Y_data = []

    for video_file in os.listdir(finished):
        video_path = os.path.join(finished, video_file)
        X_data.append(load_video_frames(video_path))
        Y_data.append(1)

    for video_file in os.listdir(unfinished):
        video_path = os.path.join(unfinished, video_file)
        X_data.append(load_video_frames(video_path))
        Y_data.append(0)

    X_data = np.array(X_data)
    Y_data = np.array(Y_data)
    return X_data, Y_data

# Veri artırma fonksiyonu
def augment_frame(frame):

    frame = tf.image.random_brightness(frame, 0.1)
    frame = tf.image.random_contrast(frame, 0.9, 1.1)

    if tf.random.uniform([]) > 0.5:
        frame = tf.image.flip_left_right(frame)

    return frame


# Tüm video üzerindeki veri artırma fonksiyonu
def augment_video(video):
    # Video boyutlarını koruyarak her bir kareye veri artırma işlemleri uygula
    augmented_video = tf.map_fn(lambda frame: augment_frame(frame), video)
    return augmented_video

# Veriyi artırma işlemini içeren tf.data pipeline
def load_data_with_augmentation(X_train, Y_train, X_val, Y_val, X_test, Y_test):

    # TRAIN dataset (augmentation VAR)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    
    train_dataset = train_dataset.map(
        lambda video, label: (augment_video(video), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    train_dataset = train_dataset.shuffle(len(X_train))
    train_dataset = train_dataset.batch(4)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)


    # VALIDATION dataset (augmentation YOK)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
    
    val_dataset = val_dataset.batch(4)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)


    # TEST dataset (augmentation YOK)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    
    test_dataset = test_dataset.batch(4)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


    return train_dataset, val_dataset, test_dataset

# Video verilerini yükleme
X_data, Y_data = load_data()

# Debug: Veri yükleme kontrolü
print(f"Yüklenen video sayısı: {len(X_data)}")
print(f"Yüklenen etiketlerin dağılımı: {np.bincount(Y_data)}")

# Verileri eğitim ve test setlerine ayırma

# %70 train, %15 val, %15 test
X_train, X_temp, Y_train, Y_temp = train_test_split(
    X_data, Y_data,
    test_size=0.30,
    random_state=42,
    stratify=Y_data
)

X_val, X_test, Y_val, Y_test = train_test_split(
    X_temp, Y_temp,
    test_size=0.50,
    random_state=42,
    stratify=Y_temp
)
# Veri artırmalı veri seti oluşturma
train_dataset, val_dataset, test_dataset = load_data_with_augmentation(
    X_train, Y_train,
    X_val, Y_val,
    X_test, Y_test
)


# Gelişmiş model yapısı
model = Sequential([
    Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', input_shape=(MAX_FRAMES, *FRAME_SIZE, 1)),
    BatchNormalization(),
    MaxPooling3D(pool_size=(1, 2, 2)),
    Dropout(0.3),
    
    Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling3D(pool_size=(1, 2, 2)),
    Dropout(0.3),
    
    Conv3D(filters=128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling3D(pool_size=(1, 2, 2)),
    Dropout(0.3),

    TimeDistributed(GlobalAveragePooling2D()),  # Flatten katmanı
    LSTM(128, return_sequences=True),  # Daha derin LSTM
    Dropout(0.5),
    
    LSTM(64),  # Ek LSTM katmanı
    Dropout(0.5),
    
    # Burada LSTM katmanı çıktı boyutu 64 oluyor.
    Dense(64, activation='relu'),  # Özellikleri yoğun katmanda çıkartma
    Dropout(0.5),
    Dense(128, activation='relu'),

    Dense(1, activation='sigmoid')  # Çıkış katmanı
])

# Model özeti
model.summary()

# Modeli derleme
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy', Precision(), Recall(), AUC()])

# Erken durdurma ve öğrenme hızını azaltma
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Modeli eğitme
history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[early_stopping])

# Eğitim ve doğrulama kaybı ile doğruluk eğrilerini çizme
plt.plot(history.history['binary_accuracy'], label='accuracy')
plt.plot(history.history['val_binary_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig("metric_results/accuracy_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# Test verisi üzerinde modelin performansını değerlendirme
Y_pred = model.predict(test_dataset)
Y_pred_classes = (Y_pred > 0.5).astype("int32")

# Confusion matrix ve diğer metrikler
confusion_mtx = confusion_matrix(Y_test, Y_pred_classes)
print("Confusion Matrix Test:")
print(confusion_mtx)


accuracy = accuracy_score(Y_test, Y_pred_classes)
precision = precision_score(Y_test, Y_pred_classes)
recall = recall_score(Y_test, Y_pred_classes)
f1 = f1_score(Y_test, Y_pred_classes)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")





# Confusion Matrix'i görselleştirme
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues", xticklabels=["Red Win", "Blue Win"], yticklabels=["Red Win", "Blue Win"])
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')
plt.title('Confusion Matrix')
plt.savefig("metric_results/cm.png", dpi=300, bbox_inches="tight")
plt.show()

model.save("model.h5")

from tensorflow.keras.models import load_model

# .h5 modelini yükle
model = load_model('model.h5')


#############################################
plt.figure(figsize=(12,16))

# Accuracy
plt.subplot(4, 1, 1)
plt.plot(history.history['binary_accuracy'], label='Train Accuracy')
plt.plot(history.history['val_binary_accuracy'], label='Val Accuracy')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Accuracy')

# Precision
plt.subplot(4, 1, 2)
plt.plot(history.history['precision'], label='Train Precision')
plt.plot(history.history['val_precision'], label='Val Precision')
plt.ylabel('Precision')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Precision')

# Recall
plt.subplot(4, 1, 3)
plt.plot(history.history['recall'], label='Train Recall')
plt.plot(history.history['val_recall'], label='Val Recall')
plt.ylabel('Recall')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Recall')

# AUC
plt.subplot(4, 1, 4)
plt.plot(history.history['auc'], label='Train AUC')
plt.plot(history.history['val_auc'], label='Val AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('AUC')

plt.tight_layout()
plt.savefig("metric_results/metrics.png", dpi=300, bbox_inches="tight")
plt.show()


from sklearn.metrics import roc_curve, auc

# ROC eğrisi için skorları al
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
roc_auc = auc(fpr, tpr)

# ROC eğrisini çiz
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig("metric_results/roc.png", dpi=300, bbox_inches="tight")
plt.show()


# # Tahminleri al
# Y_train_pred = model.predict(X_train)
# Y_test_pred = model.predict(X_test)

# # Sınıf eşikleri: 0.5 üzerinde olanlar 1
# Y_train_classes = (Y_train_pred > 0.5).astype(int)
# Y_test_classes = (Y_test_pred > 0.5).astype(int)

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(Y_test, Y_pred)

print(f"ROC-AUC: {auc:.2f}")
with open("metric_results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC-AUC: {auc:.4f}\n")
