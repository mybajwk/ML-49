import torch
import cupy as cp

if torch.cuda.is_available():
    print("CUDA tersedia!")
    print("Nama device:", torch.cuda.get_device_name(0))
else:
    print("CUDA tidak tersedia.")
    

print(cp.cuda.runtime.getDeviceCount())
print(cp.cuda.runtime.getDevice())
print(cp.cuda.runtime.runtimeGetVersion())