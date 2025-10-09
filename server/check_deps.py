try:
    import fastapi
    print('✅ FastAPI available')
except:
    print('❌ FastAPI not available')

try:
    import transformers
    print('✅ Transformers available')
except:
    print('❌ Transformers not available')

try:
    import torch
    print('✅ PyTorch available')
except:
    print('❌ PyTorch not available')

try:
    import cv2
    print('✅ OpenCV available')
except:
    print('❌ OpenCV not available')

print('Dependencies check completed!')