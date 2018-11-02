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

#define out_channel_num 6
#define out_y_dim 358
#define out_x_dim 638
#define in_y_dim 720
#define in_x_dim 1280
#define filter_size 36


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

  unsigned int i;

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
  /* (16, 16, z) (choose your z dimension) threads per block */
  /* NOTE: threads per block limit is 1024 for K80 */
  /* NOTE: if you use another GPU, check the deviceQuery */

  layer1_init_bias<<<???,???>>>(d_y, d_bias);

  /* Just in case, put a sync here */
  cudaDeviceSynchronize();

  /*********************************************
   * Layer 1, Step 2: 
   * loop over output feature maps
   ********************************************/
  /* (8, 8, z) (choose your z dimension) threads per block */
  /***********************************************
   * The layer size is not diviadable by 8 either.
   * Mask out extra threads in the kernel.
   **********************************************/  
  
  layer1_feature_maps<<<???,???>>>(d_y, d_in_layer, d_weight);

  /* Just in case, put a sync here */
  cudaDeviceSynchronize();

  /********************************************
   (14, 14, z) (choose your z dimension) threads per block
   ********************************************
   * Layer 1, Step 3: 
   * sigmoid activation function
   ********************************************/
  
  layer1_sigmoid<<<???,???>>>(d_y, d_out_layer);

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
 ********************************************/
__global__ void layer1_init_bias(float* d_y, float* d_bias) {

}

/*********************************************
 * GPU kernel
 * Layer 1, Step 2: 
 * loop over output feature maps
 ********************************************/
__global__ void layer1_feature_maps(float* d_y, unsigned char* d_in_layer, float* d_weight) {

}

/*********************************************
 * GPU kernel
 * Layer 1, Step 3: 
 * sigmoid activation function
 ********************************************/
__global__ void layer1_sigmoid(float* d_y, unsigned char* d_out_layer){


}
