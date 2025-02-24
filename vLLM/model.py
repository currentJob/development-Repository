import torch

print("Torch 버전:", torch.__version__)
print("CUDA 사용 가능 여부:", torch.cuda.is_available())
print("CUDA 버전:", torch.version.cuda)
print("cuDNN 버전:", torch.backends.cudnn.version())
print("GPU 개수:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU 이름:", torch.cuda.get_device_name(0))
    print("현재 사용 중인 장치:", torch.cuda.current_device())
