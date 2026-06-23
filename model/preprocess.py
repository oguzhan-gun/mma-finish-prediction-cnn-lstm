import numpy as np
import cv2
import os
import tensorflow as tf


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