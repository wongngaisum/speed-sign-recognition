# Speed Sign Detection and Recognition
The detection is performed in multiple layers, each consists of multiple feature maps. We will use a large input image of 1280x720 pixels, and an optimized CNN that merges the convolution layer and subsampling layer.

To compute each output element on the feature map, three major steps are performed:

1. Load the bias.

2. Perform convolution over a window.

3. Apply the sigmoid function to the result.


The ﬁle "cuda_functions.cu" consists of the empty CUDA codes for the ﬁrst convolution layer.