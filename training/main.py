import os
import kagglehub
import tensorflow as tf
import numpy as np

# ==========================================
# 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU
# ==========================================
print("Đang kiểm tra và tải dataset...")
dataset_base_path = kagglehub.dataset_download("jcoral02/inriaperson")
print(f"Dataset đã sẵn sàng tại: {dataset_base_path}")

train_dir = dataset_base_path 

BATCH_SIZE = 32
IMG_SIZE = (128, 128)

print("Đang load dữ liệu vào bộ nhớ...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    validation_split=0.2,
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

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 2. XÂY DỰNG MÔ HÌNH (ĐÃ SỬA LỖI)
# ==========================================
print("Đang xây dựng MobileNetV2...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    alpha=0.5,           
    include_top=False, 
    weights='imagenet'
)

# Đóng băng phần thân để giữ lại các đặc trưng đã học
base_model.trainable = False

model = tf.keras.Sequential([
    # CHUẨN HÓA PIXEL: Ép dải [0, 255] về [-1, 1]
    tf.keras.layers.Rescaling(1./127.5, offset=-1), 
    
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2), # Chống học vẹt
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# ==========================================
# 3. HUẤN LUYỆN (TRAINING)
# ==========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

print("Bắt đầu Training...")
model.fit(train_dataset, validation_data=validation_dataset, epochs=10)

# ==========================================
# 4. ÉP KIỂU (QUANTIZATION) SANG INT8
# ==========================================
print("Bắt đầu lượng tử hóa xuống INT8...")

def representative_data_gen():
    for input_value, _ in train_dataset.take(50): 
        # Ép kiểu float32 để TFLite đọc chuẩn xác
        yield [tf.cast(input_value, tf.float32)] 

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_quant = converter.convert()

tflite_path = 'mobilenet_v2_128_quant.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model_quant)

# ==========================================
# 5. XUẤT RA MẢNG C CHO NHÚNG BARE-METAL
# ==========================================
print("Đang xuất mảng C-Header...")
with open(tflite_path, 'rb') as f:
    tflite_content = f.read()

hex_array = ', '.join([f'0x{byte:02x}' for byte in tflite_content])

c_code = f"""#ifndef MODEL_DATA_H
#define MODEL_DATA_H

// Kích thước mảng: {len(tflite_content)} bytes
const unsigned int model_data_len = {len(tflite_content)};
const unsigned char model_data[] __attribute__((aligned(4))) = {{
    {hex_array}
}};

#endif // MODEL_DATA_H
"""

with open("model_data.h", 'w') as f:
    f.write(c_code)

print("HOÀN TẤT TOÀN BỘ QUY TRÌNH!")