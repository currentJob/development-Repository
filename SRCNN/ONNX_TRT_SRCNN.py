import os
import cv2
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Input, ReLU, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tf2onnx
import onnxruntime as ort

# ----------------------------------
# 모델 정의
# ----------------------------------
def build_advanced_srcnn():
    inputs = Input(shape=(None, None, 3))
    x = Conv2D(64, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = conv_block(x, 64, 10)

    x = Conv2D(3, 3, padding='same')(x)
    outputs = Add()([inputs, x])

    return Model(inputs, outputs)

def conv_block(x, filters, repeat):
    for _ in range(repeat):
        x = Conv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    return x

# ----------------------------------
# 데이터 전처리
# ----------------------------------
def preprocess_pair_fixed(image_path, upscale_factor):
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
    buffer_size = min(5000, images_count)

    print(f"Total images: {images_count}\nbuffer size: {buffer_size}")
    return dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ----------------------------------
# 모델 학습
# ----------------------------------
def model_train(train_dir, model_dir):
    dataset = create_fixed_dataset(train_dir, BATCH_SIZE)
    model = build_advanced_srcnn()
    model.compile(optimizer=Adam(1e-4), loss="mae")
    model.fit(dataset, epochs=EPOCHS)
    model.save(model_dir)

# ----------------------------------
# TensorFlow → ONNX 변환
# ----------------------------------
def model_convert_to_onnx(saved_model_dir, onnx_path):
    model = tf.keras.models.load_model(saved_model_dir)
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        output_path=onnx_path,
        opset=13
    )
    print(f"ONNX 모델 저장 완료: {onnx_path}")

# ----------------------------------
# ONNX → TensorRT 변환
# ----------------------------------
def convert_onnx_to_tensorrt(onnx_path, engine_path, fp16_mode=True, max_workspace_size=1 << 30):
    """
    ONNX 모델을 TensorRT 엔진으로 변환 (.engine 또는 .plan 파일 생성)
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size

    if fp16_mode:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✔ FP16 모드 활성화")
        else:
            print("⚠ FP16 지원되지 않음, FP32로 진행")

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"Parser Error {i}: {parser.get_error(i)}")
            raise RuntimeError("ONNX 모델 파싱 실패")

    # ✔ Optimization profile 설정
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name

    # 최소 / 최적 / 최대 해상도 설정
    min_shape = (1, 64, 64, 3)
    opt_shape = (1, 512, 512, 3)
    max_shape = (1, 2048, 2048, 3)

    profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)

    # 엔진 빌드
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("TensorRT 엔진 생성 실패")

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    print(f"✔ TensorRT 엔진 저장 완료: {engine_path}")

# ----------------------------------
# ONNX 추론 함수
# ----------------------------------
def load_engine(engine_path):
    """
    TensorRT 엔진 로드 함수
    """
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)

    with open(engine_path, "rb") as f:
        engine_data = f.read()
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

def infer_with_tensorrt(engine, input_image_np):
    """
    TensorRT 엔진을 통한 추론 실행
    """
    import pycuda.driver as cuda
    import pycuda.autoinit
    import tensorrt as trt

    context = engine.create_execution_context()
    input_shape = input_image_np.shape

    # 입력 / 출력 버퍼
    input_nbytes = input_image_np.nbytes
    output_nbytes = input_nbytes

    d_input = cuda.mem_alloc(input_nbytes)
    d_output = cuda.mem_alloc(output_nbytes)

    bindings = [int(d_input), int(d_output)]

    # Host → Device 복사
    cuda.memcpy_htod(d_input, input_image_np)

    # 추론 실행
    context.set_binding_shape(0, input_shape)
    context.execute_v2(bindings)

    # 결과 복사 Device → Host
    output_np = np.empty_like(input_image_np)
    cuda.memcpy_dtoh(output_np, d_output)

    return output_np

# ----------------------------------
# 업스케일 (ONNX 모델 사용)
# ----------------------------------
def upscale_image_tensorrt(image_path, engine_path):
    """
    TensorRT 엔진을 이용한 업스케일 함수
    """
    engine = load_engine(engine_path)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    lr = cv2.resize(img, (w // UPSCALE_FACTOR, h // UPSCALE_FACTOR), interpolation=cv2.INTER_CUBIC)
    lr_up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)

    input_tensor = np.expand_dims(lr_up.astype(np.float32) / 255.0, axis=0)

    output = infer_with_tensorrt(engine, input_tensor)

    sr = np.clip(output[0] * 255.0, 0, 255).astype(np.uint8)

    basename, ext = os.path.splitext(os.path.basename(image_path))
    cv2.imwrite(f"{basename}_lr_up{ext}", cv2.cvtColor(lr_up, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{basename}_sr_trt{ext}", cv2.cvtColor(sr, cv2.COLOR_RGB2BGR))

# ----------------------------------
# 설정 값
# ----------------------------------
FIXED_SIZE = (512, 512)
UPSCALE_FACTOR = 2
BATCH_SIZE = 4
EPOCHS = 20

# ----------------------------------
# 실행
# ----------------------------------
if __name__ == "__main__":
    train_img_path = "img"
    model_dir = "saved_model"
    onnx_path = "model.onnx"
    engine_path = "model.plan"
    test_img_path = "0021.png"

    # 모델 학습 및 저장
    # model_train(train_img_path, model_dir)

    # ONNX 변환
    # model_convert_to_onnx(model_dir, onnx_path)
    
    # TensorRT 엔진 변환
    # convert_onnx_to_tensorrt(onnx_path, engine_path)

    # 추론 (TensorRT)
    upscale_image_tensorrt(test_img_path, engine_path)
