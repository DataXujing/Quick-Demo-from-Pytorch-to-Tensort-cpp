# Quick-Demo-from-Pytorch-to-Tensort-cpp

 A simple and quick example shows how to convert a pytorch's model to ONNX, and then deploy it with tensorrt using c++.

 ## Pipline
  - step 1. Using Pytorch to build a simple neural network and then export to ONNX file `test.onnx`
 ```bash
 git clone https://github.com/hova88/Quick-Demo-from-Pytorch-to-Tensort-cpp.git
 cd Quick-Demo-from-Pytorch-to-Tensort-cpp
 python pytorch_to_onnx.py
 ```
The screen will shown as this:
  ```
 ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 24, 24]             260
            Conv2d-2             [-1, 20, 8, 8]           5,020
         Dropout2d-3             [-1, 20, 8, 8]               0
            Linear-4                   [-1, 50]          16,050
            Linear-5                   [-1, 10]             510
================================================================
Total params: 21,840
Trainable params: 21,840
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.06
Params size (MB): 0.08
Estimated Total Size (MB): 0.15
----------------------------------------------------------------
Dummy output:
tensor([[-2.2734, -2.2539, -2.2107, -2.3713, -2.3443, -2.4683, -2.3709, -2.2777,
         -2.2169, -2.2676]], device='cuda:0')
```

 - step 2. Initialize tensorrt engine by ONNX file, and then DO INFERENCE.
 ```bash
 mkdir build && cd build && cmake ..
 make -j6
 ./onnxTotrt
 ```
 The screen will shown as this:
 ```bash
 ----------------------------------------------------------------
Input filename:   ../test.onnx
ONNX IR version:  0.0.4
Opset version:    10
Producer name:    pytorch
Producer version: 1.3
Domain:           
Model version:    0
Doc string:       
----------------------------------------------------------------
WARNING: [TRT]/home/hova/onnx-tensorrt/onnx2trt_utils.cpp:220: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
loaded trt model , do inference
Check output:
-2.20698 -2.24908 -2.4583 -2.27005 -2.30133 -2.43457 -2.33783 -2.21271 -2.41685 -2.1832
 ```
