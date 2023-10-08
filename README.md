# Face Recognition End to End System
Welcome to the Face Recognition End to End System, a powerful solution that combines the capabilities of MTCNN and MobileFacenet models using TensorFlow. This system is designed for seamless face detection and recognition, with support for both GPU and CPU processing. Additionally, it offers features for converting models into TensorRT runtime and TensorFlow Lite format, enabling easy testing and deployment on edge devices.

## System Requirements
- python>=3.9 (Recommend 3.9.13)
- opencv-python
- numpy==1.23.1
- scipy
- tf_slim
- scikit-learn
- scikit-image
- pycuda
- tensorflow([and-cuda] optional)
## Running Guide
1. Save images of the individuals you want to recognize in face_db folder. Ensure that each image contains only one person and is named using the person's label, e.g., "Sunday.jpg."
2. cd ./face_recognition/
3. python camera_...demo.py

## Performance
System specification: Xeon E3 1240, 16gb, GTX 1070

| Run type |  GPU  |  CPU  |  TRT  |  TFLite  |
| -------- | ----- | ----- | ----- | -------- |
|   FPS    | 50-60 | 40-50 | 50-60 |  30-40   |
|   Uisng GPU (%)  |  20-25  |    ~    |  23-28  |    ~    |  
|   Uisng CPU (%)  |  15-20  |  35-40  |  20-25  |  10-15  |  


## Referencess
1. [Face recognition model: MobileFacenet](https://arxiv.org/abs/1804.07573)
2. [MobileFacenet with Tensorflow](https://github.com/sirius-ai/MobileFaceNet_TF)
3. [Face detection model: MTCNN](https://arxiv.org/abs/1604.02878)
4. [MTCNN with Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow)
5. [TensorRT runtime](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html?fbclid=IwAR21MoF2yeZWshuXywCMP97iM_TSFTdI_gboOz4tLQlJF6Exrn8Gik9UlHs)
6. [Tensorflow to TensorRT with Onnx](https://github.com/riotu-lab/tf2trt_with_onnx)
7. [Tensorflow Lite](https://www.tensorflow.org/lite/guide)
