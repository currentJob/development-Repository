import os
import cv2
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, ReLU, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.compiler.tensorrt import trt_convert as trt

def build_advanced_srcnn():
    """
    SRCNN 모델 정의
    """
    inputs = Input(shape=(FIXED_SIZE[0], FIXED_SIZE[1], 3))

    x = Conv2D(128, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    for _ in range(3):
        x = Conv2D(32, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    x = Conv2D(3, 3, padding='same')(x)
    outputs = Add()([inputs, x])

    return Model(inputs, outputs)

def preprocess_pair_fixed(image_path, upscale_factor):
    """
    저해상도-고해상도 이미지 쌍 생성
    """
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, FIXED_SIZE)
    
    _low = cv2.resize(img, (FIXED_SIZE[0] // upscale_factor, FIXED_SIZE[1] // upscale_factor), interpolation=cv2.INTER_CUBIC)
    _low = cv2.resize(_low, FIXED_SIZE, interpolation=cv2.INTER_CUBIC)
    
    _low = _low.astype(np.float32) / 255.0
    _high = img.astype(np.float32) / 255.0

    return _low, _high

def create_fixed_dataset(image_dir, batch_size):
    """
    데이터셋 구축
    """
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    images_path = []

    for ext in extensions:
        images_path.extend(glob(os.path.join(image_dir, ext)))

    def generator():
        for path in images_path:
            try:
                _low, _high = preprocess_pair_fixed(path, UPSCALE_FACTOR)
                yield _low, _high
            except:
                continue

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(FIXED_SIZE[0], FIXED_SIZE[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(FIXED_SIZE[0], FIXED_SIZE[1], 3), dtype=tf.float32),
        )
    )

    images_count = len(images_path)
    BUFFER_SIZE = min(5000, images_count)

    print(f"Total images: {images_count}\nbuffer size: {BUFFER_SIZE}")
    return dataset.shuffle(BUFFER_SIZE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def model_train(train_dir, model_dir):
    """
    모델 학습 및 저장
    """
    dataset = create_fixed_dataset(train_dir, BATCH_SIZE)

    model = build_advanced_srcnn()
    model.compile(optimizer=Adam(1e-4), loss="mae")

    model.fit(dataset, epochs=EPOCHS)
    model.save(model_dir)

def model_convert(model_dir, trt_model_dir):
    """
    모델 TRT 변환
    """
    conversion_params = trt.TrtConversionParams(
        precision_mode=trt.TrtPrecisionMode.FP16,
        max_workspace_size_bytes=1 << 25
    )
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=model_dir,
        conversion_params=conversion_params
    )
    converter.convert()
    converter.save(trt_model_dir)

def upscale_image(image_path, model_path):
    """
    원본 이미지를 업스케일
    """
    model = tf.saved_model.load(model_path)
    infer = model.signatures["serving_default"]

    _low, _ = preprocess_pair_fixed(image_path, UPSCALE_FACTOR)
    input_tensor = np.expand_dims(_low, axis=0)

    lr_up = (_low * 255.0).astype(np.uint8)
    
    output = infer(tf.constant(input_tensor))
    sr = list(output.values())[0].numpy()[0]
    sr = np.clip(sr * 255.0, 0, 255).astype(np.uint8)
    
    basename, ext = os.path.splitext(os.path.basename(image_path))
    cv2.imwrite(f"{basename}_up{ext}", cv2.cvtColor(lr_up, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{basename}_sr{ext}", cv2.cvtColor(sr, cv2.COLOR_RGB2BGR))

FIXED_SIZE = (1024, 1024)
UPSCALE_FACTOR = 4
BATCH_SIZE = 8
EPOCHS = 10

if __name__ == "__main__":
    train_img_path = "img"

    model_dir = "saved_model"
    trt_model_dir = "trt_saved_model"

    test_img_path = "test.png"

    # model_train(train_img_path, model_dir)
    # model_convert(model_dir, trt_model_dir)
    upscale_image(test_img_path, trt_model_dir)