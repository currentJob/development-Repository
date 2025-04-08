import os
import cv2
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, ReLU, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import matplotlib.pyplot as plt

# 고정 입력 크기 & 업스케일 팩터
FIXED_SIZE = (512, 512)
UPSCALE_FACTOR = 2

# 고급 SRCNN 모델 정의 (Residual 포함)
def build_advanced_srcnn():
    inputs = Input(shape=(FIXED_SIZE[0], FIXED_SIZE[1], 3))

    x = Conv2D(64, (9, 9), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    for _ in range(3):
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    x = Conv2D(3, (3, 3), padding='same')(x)
    outputs = Add()([inputs, x])

    return Model(inputs, outputs)

def preprocess_pair_fixed(image_path, upscale_factor=2):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, FIXED_SIZE)
    
    lr = cv2.resize(img, (FIXED_SIZE[0] // upscale_factor, FIXED_SIZE[1] // upscale_factor), interpolation=cv2.INTER_CUBIC)
    lr = cv2.resize(lr, FIXED_SIZE, interpolation=cv2.INTER_CUBIC)
    
    lr = lr.astype(np.float32) / 255.0
    hr = img.astype(np.float32) / 255.0
    return lr, hr

# Dataset 생성
def create_fixed_dataset(image_dir, batch_size=2):
    image_paths = glob(os.path.join(image_dir, "*.jpg"))

    def generator():
        for path in image_paths:
            try:
                lr, hr = preprocess_pair_fixed(path, UPSCALE_FACTOR)
                yield lr, hr
            except:
                continue

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(FIXED_SIZE[0], FIXED_SIZE[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(FIXED_SIZE[0], FIXED_SIZE[1], 3), dtype=tf.float32),
        )
    )
    return dataset.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 테스트용 이미지 전처리
def preprocess_test_image(image_path, upscale_factor=4):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("이미지를 불러올 수 없습니다.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, FIXED_SIZE)

    lr = cv2.resize(img, (FIXED_SIZE[0] // upscale_factor, FIXED_SIZE[1] // upscale_factor), interpolation=cv2.INTER_CUBIC)
    lr_up = cv2.resize(lr, FIXED_SIZE, interpolation=cv2.INTER_CUBIC)

    lr_up_norm = lr_up.astype(np.float32) / 255.0
    lr_up_norm = np.expand_dims(lr_up_norm, axis=0)
    return lr_up_norm, lr, img

# TensorRT 추론 및 결과 저장
def upscale_and_show_trt(image_path, infer):
    input_tensor, lr_img, hr_gt = preprocess_test_image(image_path, UPSCALE_FACTOR)
    input_tensor = tf.constant(input_tensor)

    output = infer(input_tensor)
    sr = list(output.values())[0].numpy()[0]
    sr = np.clip(sr * 255.0, 0, 255).astype(np.uint8)

    cv2.imwrite("result_lr.jpg", cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite("result_sr.jpg", cv2.cvtColor(sr, cv2.COLOR_RGB2BGR))
    cv2.imwrite("result_hr.jpg", cv2.cvtColor(hr_gt, cv2.COLOR_RGB2BGR))
    print("✅ 결과 저장 완료: result_lr.jpg / result_sr.jpg / result_hr.jpg")

# 학습 파이프라인
def train_and_convert():
    dataset = create_fixed_dataset("img")

    model = build_advanced_srcnn()
    model.compile(optimizer=Adam(1e-4), loss="mae")

    print("🔧 모델 학습 중...")
    model.fit(dataset, epochs=100)

    model.save("saved_model")

    print("🔁 TensorRT FP16 변환 중...")
    conversion_params = trt.TrtConversionParams(
        precision_mode=trt.TrtPrecisionMode.FP16,
        max_workspace_size_bytes=1 << 25
    )
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir="saved_model",
        conversion_params=conversion_params
    )
    converter.convert()
    converter.save("trt_saved_model")

    print("🚀 변환 완료, 추론 준비")

    # TRT 모델 로딩 및 추론 테스트
    trt_model = tf.saved_model.load("trt_saved_model")
    infer = trt_model.signatures["serving_default"]
    upscale_and_show_trt("image2.jpg", infer)

def load_and_infer(image_path, model_path="saved_model", use_trt=False):
    """
    저장된 모델을 불러와서 이미지 업스케일링 수행
    :param image_path: 테스트할 이미지 경로
    :param model_path: 불러올 모델 경로
    :param use_trt: True면 TensorRT 모델 로딩
    """
    print("📦 모델 로딩 중:", model_path)
    model = tf.saved_model.load(model_path)
    infer = model.signatures["serving_default"]

    # 이미지 전처리
    input_tensor, lr_img, hr_gt = preprocess_test_image(image_path, UPSCALE_FACTOR)
    input_tensor = tf.constant(input_tensor)

    # 추론
    output = infer(input_tensor)
    sr = list(output.values())[0].numpy()[0]
    sr = np.clip(sr * 255.0, 0, 255).astype(np.uint8)

    # 결과 저장
    basename = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(f"{model_path}_{basename}_lr.jpg", cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{model_path}_{basename}_sr.jpg", cv2.cvtColor(sr, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{model_path}_{basename}_gt.jpg", cv2.cvtColor(hr_gt, cv2.COLOR_RGB2BGR))
    print("✅ 결과 저장 완료:", f"{basename}_sr.jpg")

def upscale_original_image(image_path, model_path="saved_model"):
    """
    원본 이미지를 업스케일(예: x2)하여 SR 적용
    """
    print("📂 모델 로드 중:", model_path)
    model = tf.saved_model.load(model_path)
    infer = model.signatures["serving_default"]

    # 원본 이미지 로딩
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"이미지 로딩 실패: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    target_size = (w * UPSCALE_FACTOR, h * UPSCALE_FACTOR)

    # 업스케일 (Bicubic) → 모델 입력 준비
    img_up = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    img_up_norm = img_up.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(img_up_norm, axis=0)  # (1, H, W, 3)

    # 추론
    output = infer(tf.constant(input_tensor))
    sr = list(output.values())[0].numpy()[0]
    sr = np.clip(sr * 255.0, 0, 255).astype(np.uint8)

    # 저장
    basename = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(f"{basename}_bicubic_up.jpg", cv2.cvtColor(img_up, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{basename}_sr.jpg", cv2.cvtColor(sr, cv2.COLOR_RGB2BGR))
    print(f"✅ 저장 완료: {basename}_sr.jpg (업스케일 후 향상된 SR 이미지)")

# 실행
if __name__ == "__main__":
    # train_and_convert()

    upscale_original_image("image3.jpg", model_path="trt_saved_model")

    # # 일반 모델로 업스케일링
    # load_and_infer("image2.jpg", model_path="saved_model", use_trt=False)

    # # TensorRT FP16 모델로 업스케일링
    # load_and_infer("image2.jpg", model_path="trt_saved_model", use_trt=True)