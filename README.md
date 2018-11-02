# Speed Sign Detection and Recognition
The detection is performed in multiple layers, each consists of multiple feature maps. We will use a large input image of 1280x720 pixels, and an optimized CNN that merges the convolution layer and subsampling layer.

## What is CNN
> In recent years, deep learning has been used extensively in a wide range of ﬁelds. In deep learning, Convolutional Neural Networks are found to give the most accurate results in solving real world problems. CNN is used in computer vision, mainly in face recognition, scene labelling, image classiﬁcation, action recognition, human pose estimation and document analysis.

To compute each output element on the feature map, three major steps are performed:

1. Load the bias.

2. Perform convolution over a window.

3. Apply the sigmoid function to the result.

Other layers have similar structures.

## Output
```
comp3231s10@comp3231s10:~/A2$ make run
./cnn
CPU Elapsed time is 6.680574 s
number of detections = 8
detection nr 0 = 30 km/h, box pos= x 1072, y 304, confidence = 71
detection nr 1 = 30 km/h, box pos= x 1068, y 308, confidence = 96
detection nr 2 = 30 km/h, box pos= x 1072, y 308, confidence = 98
detection nr 3 = 50 km/h, box pos= x 312, y 420, confidence = 97
detection nr 4 = 50 km/h, box pos= x 316, y 420, confidence = 74
detection nr 5 = 50 km/h, box pos= x 312, y 424, confidence = 98
detection nr 6 = 50 km/h, box pos= x 316, y 424, confidence = 88
detection nr 7 = 50 km/h, box pos= x 704, y 432, confidence = 57

 ######## GPU running... ######## 
cudaMalloc success.
Time to execute layer1_feature_maps:  0.8 ms 
GPU Elapsed time is  3.6 ms 
Checking GPU results of layer 1 ...
GPU layer 1 passed.
```

1.cu and 2.cu are my implementation of the first convolution layer using GPU.
[Report](3035380875.pdf)
[Details](A2_v7.pdf)