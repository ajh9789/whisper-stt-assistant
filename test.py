
import torch
print("GPU 사용 가능?", torch.cuda.is_available())
print("현재 PyTorch CUDA 버전:", torch.version.cuda)
print("CUDA 장치 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "X")
