import os
import kagglehub
import tensorflow as tf
import numpy as np

# 1. Tải dataset (hoặc lấy từ cache ~/.cache/...)
print("Đang kiểm tra và tải dataset...")
dataset_base_path = kagglehub.dataset_download("jcoral02/inriaperson")
print(f"Dataset đã sẵn sàng tại: {dataset_base_path}")

# Lưu ý: Tùy thuộc vào cấu trúc thư mục giải nén của bộ INRIA này, 
# bạn có thể cần trỏ sâu hơn vào thư mục chứa ảnh train. 
# Ví dụ: path_to_train = os.path.join(dataset_base_path, 'Train')
# Tạm thời mình lấy thẳng dataset_base_path, bạn print thư mục ra để check nhé.
train_dir = dataset_base_path 

# 2. Cài đặt các thông số cho MobileNet (Ưu tiên ảnh nhỏ cho Edge AI)
BATCH_SIZE = 32
IMG_SIZE = (128, 128) # Dùng 128x128 hoặc 160x160 thay vì 224x224 để tiết kiệm tài nguyên FPGA

# 3. Load data vào tf.data.Dataset
print("Đang load dữ liệu vào bộ nhớ...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2, # Chia 20% làm tập validation
    subset="training",
    seed=42
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42
)

# Tối ưu hóa pipeline đọc dữ liệu để train nhanh hơn
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print("Load data hoàn tất. Sẵn sàng đưa vào MobileNet!")

# --- 1. Xây dựng Model MobileNetV2 (Alpha=0.5) ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    alpha=0.5,           # Giảm số lượng channel để siêu nhẹ cho FPGA
    include_top=False, 
    weights='imagenet'
)
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# --- 2. Training (Tóm tắt) ---
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, validation_data=validation_dataset, epochs=10)

# --- 3. Post-Training Quantization sang INT8 ---
def representative_data_gen():
    # Lấy một vài batch từ tập validation để làm mẫu ép kiểu
    for input_value, _ in validation_dataset.take(20):
        yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ép buộc output và input là INT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_quant = converter.convert()

# Lưu file tflite
with open('mobilenet_v2_128_quant.tflite', 'wb') as f:
    f.write(tflite_model_quant)