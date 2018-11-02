// CUDA runtime
#include <cuda_runtime.h>
#include <stdio.h>
// Helper functions and utilities to work with CUDA
// #include <helper_functions.h>

/**********************************************
 * Check whether we read back the same input
 * The double check is just for debug purposes.
 * We can comment it out when benchmarking the time.
 **********************************************/
#define GPU_DEBUG


/*
  Define all constant variavle below with a REASONABLE name
*/

#define out_channel_num 6 // number of feature maps
#define out_y_dim 358 // height of output map
#define out_x_dim 638 // width of output map
#define in_y_dim 720  // height of input map
#define in_x_dim 1280 // width of output map
#define conv_window_y 6 // height of convolution window
#define conv_window_x 6 // width of convolution window
#define filter_size (conv_window_y * conv_window_x) // size of convolution window
#define stride 2  // stride of layer
#define init_bias_thread_x 16 // thread x dimension of init_bias
#define init_bias_thread_y 16 // thread y dimension of init_bias
#define init_bias_thread_z 2 // thread z dimension of init_bias
#define feature_maps_thread_x 8 // thread x dimension of feature_maps
#define feature_maps_thread_y 8 // thread y dimension of feature_maps
#define feature_maps_thread_z 8 // thread z dimension of feature_maps
#define sigmoid_thread_x 14 // thread x dimension of sigmoid
#define sigmoid_thread_y 14 // thread y dimension of sigmoid
#define sigmoid_thread_z 2 // thread z dimension of sigmoid

/******************************************
 * Device function declaration
 *****************************************/
__global__ void layer1_init_bias(float* d_y, float* d_bias);
__global__ void layer1_feature_maps(float* d_y, unsigned char* d_in_layer, float* d_weight);
__global__ void layer1_sigmoid(float* d_y, unsigned char* d_out_layer);

/************************************************************************************
 * Input   : input image, pointer to output result, coefficients bias and weights
 * Output  : neuron outputs of the feature maps represented as an image
 * Procedure: perform feed forward computation through the feature extraction layers
     *******************************************************************************/
void cuda_convolution_layer1(unsigned char in_layer[], unsigned char out_layer[],
           const float bias[], const float weight[]) {

  /*********************************
   * allocate device memory on GPU
   *********************************/

  unsigned int size_y = out_channel_num*out_y_dim*out_x_dim;
  unsigned int mem_size_y = sizeof(float) * size_y;
  float *d_y;

  unsigned int size_bias = out_channel_num;
  unsigned int mem_size_bias = sizeof(float) * size_bias;
  float *d_bias;

  unsigned int size_weight = out_channel_num*filter_size;
  unsigned int mem_size_weight = sizeof(float) * size_weight;
  float *d_weight;

  unsigned int size_in_layer = in_y_dim*in_x_dim;
  unsigned int mem_size_in_layer = sizeof(unsigned char) * size_in_layer;
  unsigned char *d_in_layer;

  unsigned int size_out_layer = out_channel_num*out_y_dim*out_x_dim;
  unsigned int mem_size_out_layer = sizeof(unsigned char) * size_out_layer;
  unsigned char *d_out_layer;

  cudaError_t error;


  /********************************
   * Allocate device memory on GPU.
   * Check the first cudaMalloc error,
   * in case GPU is busy.
   ********************************/
  error = cudaMalloc((void **) &d_y, mem_size_y);
  /* Check the error code of the first CUDA API call */
  if (error != cudaSuccess){
    printf("cudaMalloc returned error code %d, line(%d)\n", error, __LINE__);
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }else{
    printf("cudaMalloc success.\n");
  }
  /* if no error for the first cudaMalloc, continue other cudaMalloc */
  error = cudaMalloc((void **) &d_in_layer, mem_size_in_layer);
  error = cudaMalloc((void **) &d_bias, mem_size_bias);
  error = cudaMalloc((void **) &d_weight, mem_size_weight);
  error = cudaMalloc((void **) &d_out_layer, mem_size_out_layer);

  /*********************************************
   * copy data from host (CPU) to device (GPU)
   ********************************************/
  error = cudaMemcpy(d_in_layer, in_layer, mem_size_in_layer, cudaMemcpyHostToDevice);
  error = cudaMemcpy(d_bias, bias, mem_size_bias, cudaMemcpyHostToDevice);
  error = cudaMemcpy(d_weight, weight, mem_size_weight, cudaMemcpyHostToDevice);

  /* Synchronize all the cudaMemcpy API before doing the actual computation */
  cudaDeviceSynchronize();

  /*********************************************
   * Layer 1, Step 1: 
   * init values of feature maps at bias value 
   ********************************************/
  /* 16*16*z(choose the correct z dimension) threads per block */
  /* NOTE: threads per block limit is 1024 for K80 */
  /* NOTE: if you use another GPU, check the deviceQuery */

  dim3 threadsPerBlock = dim3(init_bias_thread_x, init_bias_thread_y, init_bias_thread_z);
  dim3 blocksPerGrid = dim3((out_x_dim + init_bias_thread_x - 1) / init_bias_thread_x, 
          (out_y_dim + init_bias_thread_y - 1) / init_bias_thread_y, 
          (out_channel_num + init_bias_thread_z - 1) / init_bias_thread_z);
  layer1_init_bias<<<blocksPerGrid, threadsPerBlock>>>(d_y, d_bias);

  /* Just in case, put a sync here */
  cudaDeviceSynchronize();

  /*********************************************
   * Layer 1, Step 2: 
   * loop over output feature maps
   ********************************************/
  /* 8*8*z(choose the correct z dimension) threads per block */
  /***********************************************
   * The layer size is not diviadable by 8 either.
   * Mask out extra threads in the kernel.
   **********************************************/  

  threadsPerBlock = dim3(feature_maps_thread_x, feature_maps_thread_y, feature_maps_thread_z);
  blocksPerGrid = dim3((out_x_dim + feature_maps_thread_x - 1) / feature_maps_thread_x, 
          (out_y_dim + feature_maps_thread_y - 1) / feature_maps_thread_y, 
          (out_channel_num + feature_maps_thread_z - 1) / feature_maps_thread_z);

  // record time to execute layer1_feature_maps
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  layer1_feature_maps<<<blocksPerGrid, threadsPerBlock>>>(d_y, d_in_layer, d_weight);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("Time to execute layer1_feature_maps:  %3.1f ms \n", time);
  
  /* Just in case, put a sync here */
  cudaDeviceSynchronize();

  /********************************************
   14*14*z(choose the correct z dimension) threads per block
   ********************************************
   * Layer 1, Step 3: 
   * sigmoid activation function
   ********************************************/

  threadsPerBlock = dim3(sigmoid_thread_x, sigmoid_thread_y, sigmoid_thread_z);
  blocksPerGrid = dim3((out_x_dim + sigmoid_thread_x - 1) / sigmoid_thread_x, 
          (out_y_dim + sigmoid_thread_y - 1) / sigmoid_thread_y, 
          (out_channel_num + sigmoid_thread_z - 1) / sigmoid_thread_z);
  layer1_sigmoid<<<blocksPerGrid, threadsPerBlock>>>(d_y, d_out_layer);

  /* Just in case, put a sync here */
  cudaDeviceSynchronize();

  /* Read back the output from device (GPU) to host (CPU) */
  error = cudaMemcpy(out_layer, d_out_layer, mem_size_out_layer, cudaMemcpyDeviceToHost);


  /* Just in case, put a sync here */
  cudaDeviceSynchronize();

  /* release device memory */
  cudaFree(d_y);
  cudaFree(d_in_layer);
  cudaFree(d_bias);
  cudaFree(d_weight);
  cudaFree(d_out_layer);

}


/*********************************************
 * GPU kernel
 * Layer 1, Step 1: 
 * init values of feature maps at bias value
 * 16*16*z(choose the correct z dimension) threads per block
 ********************************************/
__global__ void layer1_init_bias(float* d_y, float* d_bias) {

  int col = threadIdx.x + blockIdx.x * init_bias_thread_x;
  int row = threadIdx.y + blockIdx.y * init_bias_thread_y;
  int depth = threadIdx.z + blockIdx.z * init_bias_thread_z;

  if (row < out_y_dim && col < out_x_dim && depth < out_channel_num)  // prevent out of bound access
    d_y[depth * out_y_dim * out_x_dim + row * out_x_dim + col] = d_bias[depth]; // load the bias

}

/*********************************************
 * GPU kernel
 * Layer 1, Step 2: 
 * loop over output feature maps
 * 8*8*z(choose the correct z dimension) threads per block
 ********************************************/
__global__ void layer1_feature_maps(float* d_y, unsigned char* d_in_layer, float* d_weight) {

  int col = threadIdx.x + blockIdx.x * feature_maps_thread_x;
  int row = threadIdx.y + blockIdx.y * feature_maps_thread_y;
  int depth = threadIdx.z + blockIdx.z * feature_maps_thread_z;

  // cache d_in_layer
  __shared__ unsigned char in_layer[feature_maps_thread_y * stride + conv_window_y][feature_maps_thread_x * stride + conv_window_x];

  // process [0, feature_maps_thread_y * stride - 1][0, feature_maps_thread_x * stride + conv_window_x - 1]
  for (int i = 0; i < stride; i++) 
      in_layer[threadIdx.y * stride + i][threadIdx.x * stride + depth] = 
        d_in_layer[(row * stride + i) * in_x_dim + col * stride + depth];

  // process [feature_maps_thread_y * stride, feature_maps_thread_y * stride + conv_window_y - 1][0, feature_maps_thread_x * stride - 1]
  if (threadIdx.y == 0 && depth < conv_window_y)
    for (int i = 0; i < stride; i++) {
      in_layer[feature_maps_thread_y * stride + depth][threadIdx.x * stride + i] = 
        d_in_layer[((row + feature_maps_thread_y) * stride + depth) * in_x_dim + col * stride + i];
    }

  // process [feature_maps_thread_y * stride, feature_maps_thread_y * stride + conv_window_y - 1][feature_maps_thread_x * stride, feature_maps_thread_x * stride + conv_window_x - 1]
  if (threadIdx.x < conv_window_x && threadIdx.y == 0 && depth < conv_window_y)
    in_layer[feature_maps_thread_y * stride + depth][feature_maps_thread_x * stride + threadIdx.x] = 
      d_in_layer[((row + feature_maps_thread_y) * stride + depth) * in_x_dim + (col - threadIdx.x + feature_maps_thread_x) * stride + threadIdx.x];

  // cache d_weight
  __shared__ float weight[out_channel_num][filter_size];
  if (threadIdx.y < out_y_dim && threadIdx.x < out_x_dim && depth < out_channel_num)
    weight[depth][threadIdx.y * conv_window_x + threadIdx.x] = d_weight[depth * filter_size + threadIdx.y * conv_window_x + threadIdx.x];

  __syncthreads();

  if (row < out_y_dim && col < out_x_dim && depth < out_channel_num)  // prevent out of bound access
    for (int k = 0; k < conv_window_y; k++)  // loop over convolution window (row)
      for (int l = 0; l < conv_window_x; l++)  // loop over convolution window (column)
        // perform convolution over a window
        d_y[depth * out_y_dim * out_x_dim + row * out_x_dim + col] += 
          in_layer[threadIdx.y * stride + k][threadIdx.x * stride + l] * weight[depth][k * conv_window_x + l];

}

/*********************************************
 * GPU kernel
 * Layer 1, Step 3: 
 * sigmoid activation function
 * 14*14*z(choose the correct z dimension) threads per block
 ********************************************/
__global__ void layer1_sigmoid(float* d_y, unsigned char* d_out_layer){

  int col = threadIdx.x + blockIdx.x * sigmoid_thread_x;
  int row = threadIdx.y + blockIdx.y * sigmoid_thread_y;
  int depth = threadIdx.z + blockIdx.z * sigmoid_thread_z;
  int idx = depth * out_y_dim * out_x_dim + row * out_x_dim + col;  // index in the grid

  if (row < out_y_dim && col < out_x_dim && depth < out_channel_num)
    d_out_layer[idx] = (unsigned char)(255.999f / (1 + expf(-d_y[idx] / 256))); // apply the sigmoid function to the result

}