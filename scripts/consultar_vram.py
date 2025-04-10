import torch

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Memória total (MB):", round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 2))
else:
    print("CUDA não está disponível.")
