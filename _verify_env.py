import sys
print(f'Python: {sys.version.split()[0]}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

import xformers
print(f'xformers: {xformers.__version__}')

from depth_anything_3.api import DepthAnything3
print('DA3: OK')

import transformers
print(f'transformers: {transformers.__version__}')

print('\nAll good!')
