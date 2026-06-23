import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.layers import GlobalAveragePooling2D

from preprocess import load_data, load_data_with_augmentation, MAX_FRAMES, FRAME_SIZE
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dense, Dropout, BatchNormalization,LSTM, TimeDistributed

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
history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[])

# Eğitim ve doğrulama kaybı ile doğruluk eğrilerini çizme
plt.plot(history.history['binary_accuracy'], label='accuracy')
plt.plot(history.history['val_binary_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Test verisi üzerinde modelin performansını değerlendirme
Y_pred = model.predict(test_dataset)
Y_pred_classes = (Y_pred > 0.5).astype("int32")



from metrics import metric_calc

metric_calc(Y_test, Y_pred, Y_pred_classes)
