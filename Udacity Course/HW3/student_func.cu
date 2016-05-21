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

#define BLOCKSIZE 256

#define NUM_BANKS 16  
#define LOG_NUM_BANKS 4  
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

__global__ void maxReduce(const float *d_logLuminance, float *d_max_out, int size) {
	
	//shared memory, size allocated in kernel call
	extern __shared__ float s_data[];

	//get ids
	int tid = threadIdx.x;
	int mId = tid + (blockDim.x *blockIdx.x);
	//store in shared memory.
	s_data[tid] = (mId < size) ? d_logLuminance[mId] : -FLT_MAX;

	//make sure shared memory is loaded. 
	__syncthreads();

	//sequential addressing and then performing reduction between two halfs of the block size. 
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
		//if thread index is < block dimension, then perform our operation.
		if (tid < s) {
			s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
		}
		__syncthreads();
	}

	if (tid == 0) {
		d_max_out[blockIdx.x] = s_data[0];
	}
}

__global__ void minReduce(const float *d_logLuminance, float *d_min_out, int size) {

	//shared memory, size allocated in kernel call
	extern __shared__ float s_data[];

	//get ids
	int tid = threadIdx.x;
	int mId = tid + (blockDim.x *blockIdx.x);
	//store in shared memory.
	s_data[tid] = (mId < size) ? d_logLuminance[mId] : FLT_MAX;

	//make sure all the shared memory is loaded. 
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
		if (tid < s) {
			float temp = s_data[tid];
			s_data[tid] = fminf(temp, s_data[tid + s]);
		}
		__syncthreads();
	}
	
	if (tid == 0) {
		d_min_out[blockIdx.x] = s_data[0];
	}
}

__global__ void histogram(const float* const d_logLuminance, unsigned int* d_hist, const float min, const float max, const int numBins, const int size) {
	float range = max - min;
	//compute the histgram
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	if (id < size) {
		float data = d_logLuminance[id];
		int bin = ((data - min) / (range))* numBins;
		if (bin < numBins) {
			atomicAdd(&d_hist[bin], 1);
		}
	}
	
}

__global__ void exclusiveScan(unsigned int* const g_hist, unsigned int* o_data, int size) {
	
	extern __shared__ unsigned int temp[];// allocated on invocation
	int thid = threadIdx.x;

	int offset = 1;
	int n = size;
	int ai = thid;
	int bi = thid + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(ai);
	temp[ai + bankOffsetA] = g_hist[ai];
	temp[bi + bankOffsetB] = g_hist[bi];

	for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (thid == 0) { temp[n-1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; }
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2 * thid + 1) - 1;
			int bi = offset*(2 * thid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			unsigned int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	o_data[ai] = temp[ai + bankOffsetA];
	o_data[bi] = temp[bi + bankOffsetB];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
	/*Here are the steps you need to implement
	1) find the minimum and maximum value in the input logLuminance channel
	store in min_logLum and max_logLum
	2) subtract them to find the range
	3) generate a histogram of all the values in the logLuminance channel using
	the formula: bin = (lum[i] - lumMin) / lumRange * numBins
	4) Perform an exclusive scan (prefix sum) on the histogram to get
	the cumulative distribution of luminance values (this should go in the
	incoming d_cdf pointer which already has been allocated for you)       */

	int N = numCols * numRows;

	//step 1, find min and max. 
	const int maxThreadsPerBlock = 1024;
	int threads = (N < BLOCKSIZE) ? nextPow2(N) : BLOCKSIZE;
	int blocks = (N + threads - 1) / threads;

	float *d_min_out;
	float *d_max_out;
	checkCudaErrors(cudaMalloc((void**)&d_min_out, sizeof(float)*numCols*numRows));
	checkCudaErrors(cudaMalloc((void**)&d_max_out, sizeof(float)*numCols*numRows));

	//allocate shared memory.
	int size = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);

	minReduce << <blocks, threads, size >> >(d_logLuminance, d_min_out, N);
	maxReduce << < blocks, threads, size >> > (d_logLuminance, d_max_out, N);

	float *h_min_out = (float *)malloc(sizeof(float)*numRows*numCols);
	float *h_max_out = (float *)malloc(sizeof(float)*numRows*numCols);

	checkCudaErrors(cudaMemcpy(h_min_out, d_min_out, sizeof(float)*numCols*numRows, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_max_out, d_max_out, sizeof(float)*numCols*numRows, cudaMemcpyDeviceToHost));

	//perform final part of min and max on the CPU
	float min = FLT_MAX;
	float max = -FLT_MAX;
	for (int i = 0; i < N; i++) {
		min = fminf(h_min_out[i], min);
		max = fmaxf(h_max_out[i], max);
	}
	//set min and max. 
	min_logLum = min;
	max_logLum = max;	

	//now need to generate a histogram. 
	unsigned int * h_hist = (unsigned int*)malloc(sizeof(unsigned int)*numBins);
	memset(h_hist, 0, sizeof(unsigned int) * numBins);
	//allocated device memory.
	unsigned int *d_hist;
	checkCudaErrors(cudaMalloc((void**)&d_hist, sizeof(unsigned int) * numBins));
	checkCudaErrors(cudaMemcpy(d_hist, h_hist, sizeof(unsigned int)*numBins, cudaMemcpyHostToDevice));

	//generate the histogram. 
	histogram <<<blocks, threads >>>(d_logLuminance, d_hist, min_logLum, max_logLum, numBins, N);
	//copy results back. 
	checkCudaErrors(cudaMemcpy(h_hist, d_hist, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));

	//allocate shared memory. 
	size_t shMemSize = 2 * numBins *sizeof(unsigned int);
	exclusiveScan <<<1, numBins, shMemSize >> >(d_hist, d_cdf, numBins);
}
