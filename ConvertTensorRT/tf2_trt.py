

import tensorflow as tf
import numpy as np
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import os

SAVED_MODEL_DIR="../face_recognition/models/mobilefacenet_model/MobileFaceNet_9925_9680.pb"

input_saved_model_dir = os.path.join(os.getcwd(), 'face_recognition', 'models', 'mobilefacenet')
output_saved_model_dir = '../face_recognition/models/tensorRT/'
num_runs = 1

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(max_workspace_size_bytes=(1<<32))
conversion_params = conversion_params._replace(precision_mode="FP16")
# conversion_params = conversion_params._replace(maximum_cached_engiens=100)

converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir,conversion_params=conversion_params)
converter.convert()

def my_input_fn():
    for _ in range(num_runs):
        inp1 = np.random.normal(size=(1, 1, 112, 112, 3)).astype(np.uint8)
        yield inp1
        
converter.build(input_fn=my_input_fn)
converter.save(output_saved_model_dir)