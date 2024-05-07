
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Windows.h>
#include <stdlib.h>
#include <time.h>

#define DIM 768

#define SMEMDIM 100 
#define N 1000704


__global__ void plusOne(unsigned char *a, __int64 numElements, unsigned long skip)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		unsigned char temp = a[i] + 1;
		int index = i / skip;
		if (i % skip == 0)
			a[index] = temp;
		//printf("%hhu\n", a[i]);
	}
}

// Cast a data to a double and use the window data if it exists
__global__ void byteToDouble(unsigned char* in, double* window, double* out, __int64 numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		if (window)
		{
			out[i] = (double)in[i] * window[i];
		}
		else
		{
			out[i] = (double)in[i];
		}
	}
}



__global__ void plusOneShort1(short* a, __int64 numElements, unsigned long skip)
{
	__shared__ int smem[SMEMDIM];
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
#pragma unroll
	for (int i = index; i < numElements / 4; i += stride)
	{
		a[i * 4] = (a[i * 4] - a[i * 4 + 2]);
		a[i * 4 + 1] = (a[i * 4 + 1] - a[i * 4 + 3]);
	}

}

__global__ void multiplyKernel(short* a, int* dev_a,  __int64 numElements)
{
	__shared__ int smem[SMEMDIM];
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
#pragma unroll
	for (int i = index; i < numElements /4; i+=stride)
	{
		dev_a[i] = (int)a[i * 4 ] * (int)a[i * 4 + 1];
	}

}

__global__ void sumKernel(int* dev_a, int* d_accTemp, __int64 numElements)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
#pragma unroll
	for (int i = index; i < numElements ; i+=stride) 
	{
		atomicAdd(d_accTemp, (int)dev_a[i]);
	}
}
__inline__ __device__ int warpReduce(int mySum) {
	mySum += __shfl_xor(mySum, 16);
	mySum += __shfl_xor(mySum, 8);
	mySum += __shfl_xor(mySum, 4);
	mySum += __shfl_xor(mySum, 2);
	mySum += __shfl_xor(mySum, 1);
	return mySum;
}

__global__ void reduceShfl(int* g_idata, int* g_odata,
	unsigned int n)
{
	// shared memory for each warp sum
	__shared__ int smem[SMEMDIM];

	// boundary check   
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;

	// read from global memory
	int mySum = g_idata[idx];

	// calculate lane index and warp index
	int laneIdx = threadIdx.x % warpSize;
	int warpIdx = threadIdx.x / warpSize;

	// block-wide warp reduce 
	mySum = warpReduce(mySum);

	// save warp sum to shared memory
	if (laneIdx == 0) smem[warpIdx] = mySum;

	// block synchronization
	__syncthreads();

	// last warp reduce
	mySum = (threadIdx.x < SMEMDIM) ? smem[laneIdx] : 0;
	if (warpIdx == 0) mySum = warpReduce(mySum);

	// write result for this block to global mem
	if (threadIdx.x == 0) atomicAdd(g_odata,mySum);
}


// Helper function for using CUDA.

extern "C" cudaError_t GPU_Equation_PlusOne(void* a, unsigned long skip, unsigned long sample_size, __int64 size, int blocks, int threads, int u32LoopCount)
{
	cudaError_t cudaStatus = cudaSuccess;

	int deviceId;
	int numberOfSMs;
	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	blocks = numberOfSMs * 32;
	threads = 768;
	time_t t = time(NULL);
	struct tm* tm = localtime(&t);
	char s[64];
	size_t ret = strftime(s, sizeof(s), "%c", tm);

	int CPUresult = 1 ;
	int CheckRaw = 0;
	int AnalysisFile = 1;

	FILE* fptr;
	if (AnalysisFile == 1) {
		fptr = fopen("Analysis.txt", "a");
	}

	// Launch a kernel on the GPU with one thread for each element.

	int* dev_a, * d_accTemp, * d_accTemp2;
	int h_accTemp = 0;
	int h_accTemp2 = 0;

	short* h_dev_a = (short*)malloc(size * sizeof(short));
	short* h_dev_a2 = (short*)malloc(size * sizeof(short));


	//cudaStatus = cudaDeviceSynchronize();
	
	cudaStatus = cudaMalloc((void**)&dev_a, size / 4 * sizeof(int));

	//cudaStatus = cudaMalloc((void**)&temp_dev_a, size / 2 * sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_accTemp, 1 * sizeof(int));
	
	cudaStatus = cudaMalloc((void**)&d_accTemp2, 1 * sizeof(int));
	int* h_odata = (int*)malloc(1 * sizeof(int));
	int gpu_sum = 0;

	cudaStatus = cudaMemcpy(h_dev_a, a, size * sizeof(short), cudaMemcpyDeviceToHost);


	if (0 == blocks)
	{
		blocks = (int)(size + threads - 1) / threads;
	}

	if (1 == sample_size)
	{
		plusOne << <blocks, threads>> > ((unsigned char*)a, size, skip);
	}
	else
	{
		plusOneShort1 << <1536, 768 >> > ((short*)a, size, skip);

		cudaStatus = cudaDeviceSynchronize();
	}



	multiplyKernel << <blocks , threads>> > ((short*)a, (int*) dev_a, size);



	reduceShfl << <blocks, threads >> > (dev_a, (int*) d_accTemp2, N );
	cudaMemcpy(h_odata, d_accTemp2, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "deviceReduceKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	

	cudaFree(dev_a);

	cudaFree(d_accTemp);
	cudaFree(d_accTemp2);

	cudaStatus = cudaDeviceSynchronize();
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaMemcpy(h_dev_a2, a, size * sizeof(short), cudaMemcpyDeviceToHost);

	if (CPUresult == 1) {
		for (int i = 0; i < size; i++) {
			if (i % 4 == 0 && i < size - 2)
			{
				//// Write to local disk
				if (1 == CheckRaw) {
					if (h_dev_a[i] - h_dev_a[i + 2] != h_dev_a2[i])
					{
						h_dev_a[i] = h_dev_a[i] - h_dev_a[i + 2];
						h_dev_a[i + 1] = h_dev_a[i + 1] - h_dev_a[i + 3];
						printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", i, h_dev_a2[i], h_dev_a[i], h_dev_a2[i + 1], h_dev_a[i + 1], h_dev_a2[i + 2], h_dev_a[i + 2], h_dev_a2[i + 3], h_dev_a[i + 3]);

						//h_dev_a[i] = h_dev_a[i] * h_dev_a[i + 1];
						////printf("%d\t%d\t%d\n", i, h_dev_a[i], h_dev_a2[i / 4]);
						//h_dev_a[i + 1] = 0;
					}
				}
				else {
					h_dev_a[i] = h_dev_a[i] - h_dev_a[i + 2];
					h_dev_a[i + 1] = h_dev_a[i + 1] - h_dev_a[i + 3];
					h_dev_a[i + 2] = 0;
					h_dev_a[i + 3] = 0;
					h_dev_a[i] = h_dev_a[i] * h_dev_a[i + 1];
					h_dev_a[i + 1] = 0;
					h_accTemp2 += h_dev_a[i];
				}
			}
		}
	}



	if (CheckRaw != 1) {
		if (CPUresult == 1) {
			if (AnalysisFile == 1) {
				fprintf(fptr, "%d\t%lli\t%lld\n", u32LoopCount, h_accTemp2, h_odata[0]);
				fclose(fptr);
			}
		}
		else {
			if (AnalysisFile == 1) {
				fprintf(fptr, "%d\t%lld\t%lld\n", u32LoopCount, h_accTemp2, h_odata[0]);
				fclose(fptr);
			}
		}
	}




	fclose(fptr);
	// Don't forget to free allocated memory on both host and device
	if (CPUresult == 1) {
		free(h_dev_a);
		//free(h_dev_a2);
}
	free(h_odata);



	return cudaStatus;
}
