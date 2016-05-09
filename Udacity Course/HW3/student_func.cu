/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "device_atomic_functions.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif

__global__ void maxReduce(const float *d_logLuminance, float *d_max_out) {
	
	//shared memory, size allocated in kernel call
	extern __shared__ float s_data[];

	//get ids
	int tid = threadIdx.x;
	int mId = tid + (blockDim.x *blockIdx.x);
	//store in shared memory.
	s_data[tid] = d_logLuminance[tid];
	__syncthreads();

	//sequential addressing and then performing reduction between two halfs of the block size. 
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
		//if thread index is < block dimension, then perform our operation.
		if (tid < s) {
			s_data[tid] = max(s_data[tid], s_data[tid + s]);
		}
		__syncthreads();
	}

	if (tid == 0) {
		d_max_out[blockIdx.x] = s_data[0];
	}
}

__global__ void minReduce(const float *d_logLuminance, float *d_min_out) {

	//shared memory, size allocated in kernel call
	extern __shared__ float s_data[];

	//get ids
	int tid = threadIdx.x;
	int mId = tid + (blockDim.x *blockIdx.x);
	//store in shared memory.
	s_data[tid] = d_logLuminance[tid];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
		if (tid < s) {
			s_data[tid] = min(s_data[tid], s_data[tid + s]);
		}
		__syncthreads();
	}
	
	if (tid == 0) {
		d_min_out[blockIdx.x] = s_data[0];
	}
}

__global__ void histogram(const float* const d_logLuminance, const int* d_hist, const float min, const float max, const int numBins) {
	float range = max - min;
	//compute the histgram
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	float data = d_logLuminance[id];
	int bin = ((data - min) / (range))* numBins;
	atomicAdd(&(d_hist[bin]), 1);
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{

	//TODO
	/*Here are the steps you need to implement
	1) find the minimum and maximum value in the input logLuminance channel
	store in min_logLum and max_logLum
	2) subtract them to find the range
	3) generate a histogram of all the values in the logLuminance channel using
	the formula: bin = (lum[i] - lumMin) / lumRange * numBins
	4) Perform an exclusive scan (prefix sum) on the histogram to get
	the cumulative distribution of luminance values (this should go in the
	incoming d_cdf pointer which already has been allocated for you)       */

	//step 1, find min and max. 
	const int maxThreadsPerBlock = 1024;
	int threads = maxThreadsPerBlock;
	int blocks = (numRows*numCols) / maxThreadsPerBlock;

	float *d_min_out;
	float *d_max_out;
	float *d_min_int, *d_max_int;
	checkCudaErrors(cudaMalloc((void**)&d_min_int, sizeof(float)*numCols*numRows));
	checkCudaErrors(cudaMalloc((void**)&d_max_int, sizeof(float)*numCols*numRows));
	checkCudaErrors(cudaMalloc((void**)&d_min_out, sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_max_out, sizeof(float)));

	//allocate shared memory.
	size_t size = sizeof(float) * numRows * numCols; //shared memory size. 

	minReduce<<<blocks, threads, size>>>(d_logLuminance, d_min_int);
	maxReduce<<< blocks, threads, size >> > (d_logLuminance, d_max_int);

	//reduce the last block
	threads = blocks;
	blocks = 1;
	minReduce <<<blocks, threads, size >> >(d_min_int, d_min_out);
	maxReduce <<<blocks, threads, size >> >(d_max_int, d_max_out);
	//now have min and max. 
	//copy result to output floats.
	checkCudaErrors(cudaMemcpy(&min_logLum, d_min_out, sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_logLum, d_max_out, sizeof(float), cudaMemcpyDeviceToHost));

	//now need to generate a histogram. 
	int * h_hist = new int[numBins];

	//allocated device memory.
	int *d_hist;
	checkCudaErrors(cudaMalloc((void**)d_hist, sizeof(int) * numBins));
	threads = maxThreadsPerBlock;
	blocks = (numRows*numCols) / maxThreadsPerBlock;

	//generate the histogram. 
	histogram <<<blocks, threads >> >(d_logLuminance, d_hist, min_logLum, max_logLum, numBins);
	checkCudaErrors(cudaMemcpy(h_hist, d_hist, sizeof(int) * numBins,cudaMemcpyDeviceToHost));

}
