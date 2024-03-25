
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Windows.h>
#include <stdlib.h>
#include <time.h>

#define DIM 768
#define SMEMDIM 24 


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

	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
#pragma unroll
	for (int i = index; i < numElements / 16; i += stride)
	{
		a[i * 4] = (a[i * 4] - a[i * 4 + 2]);
	}

}


__global__ void plusOneShort2(short* a, __int64 numElements, unsigned long skip)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
#pragma unroll
	for (int i = index; i < numElements / 16; i += stride)
	{
		a[i*4 + 1] = (a[i*4 + 1] - a[i*4 + 3]);
	}
}
__global__ void plusOneShort3(short* a, __int64 numElements, unsigned long skip)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
#pragma unroll
	for (int i = index ;  i < numElements /16 ; i += stride)
	{
		a[i * 4] = (a[i * 4] - a[i * 4 + 2]);
	}
}


__global__ void plusOneShort4(short* a, __int64 numElements, unsigned long skip)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
#pragma unroll
	for (int i = index; i < numElements / 16; i += stride)
	{
		a[i*4 + 1] = (a[i*4 + 1] - a[i*4 + 3]);
	}
}

__global__ void plusOneShort5(short* a, __int64 numElements, unsigned long skip)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
#pragma unroll
	for (int i = index; i < numElements / 16; i += stride)
	{
		a[i*4] = (a[i*4] - a[i*4 + 2]);
	}
}


__global__ void plusOneShort6(short* a, __int64 numElements, unsigned long skip)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
#pragma unroll
	for (int i = index; i < numElements / 16; i += stride)
	{
		a[i*4 + 1] = (a[i*4 + 1] - a[i *4+ 3]);
	}
}
__global__ void plusOneShort7(short* a, __int64 numElements, unsigned long skip)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
#pragma unroll
	for (int i = index; i < numElements / 16; i += stride)
	{
		a[i*4] = (a[i*4] - a[i*4 + 2]);
	}
}


__global__ void plusOneShort8(short* a, __int64 numElements, unsigned long skip)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
#pragma unroll
	for (int i = index; i < numElements / 16; i += stride)
	{
		a[i*4 + 1] = (a[i*4 + 1] - a[i*4 + 3]);
	}
}

__global__ void multiplyKernel(short* a, int* dev_a,  __int64 numElements)
{
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
	/*int sum = int(0);
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;	i < numElements;i += blockDim.x * gridDim.x) {
		sum += dev_a[i];
	}
	sum = warpReduceSum(sum);
	if (threadIdx.x == 0)
	{
		atomicAdd(d_accTemp, sum);
	}
*/


	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
#pragma unroll
	for (int i = index; i < numElements ; i+=stride) 
	{
		atomicAdd(d_accTemp, (int)dev_a[i]);
	}
}

__global__ void reduceSmemUnrollShfl(int* g_idata, int* g_odata,
	unsigned int n)
{
	// static shared memory
	__shared__ int smem[DIM];

	// set thread ID
	unsigned int tid = threadIdx.x;

	// global index
	unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

	// unrolling 4 blocks
	int localSum = 0;

	if (idx + 3 * blockDim.x < n)
	{
		int a1 = g_idata[idx];
		int a2 = g_idata[idx + blockDim.x];
		int a3 = g_idata[idx + 2 * blockDim.x];
		int a4 = g_idata[idx + 3 * blockDim.x];
		localSum = a1 + a2 + a3 + a4;
	}

	smem[tid] = localSum;
	__syncthreads();

	// in-place reduction in shared memory
	if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
	__syncthreads();
	if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
	__syncthreads();
	if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
	__syncthreads();
	if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid + 64];
	__syncthreads();
	if (blockDim.x >= 64 && tid < 32) smem[tid] += smem[tid + 32];
	__syncthreads();

	// unrolling warp
	localSum = smem[tid];
	if (tid < 32)
	{
		localSum += __shfl_xor(localSum, 16);
		localSum += __shfl_xor(localSum, 8);
		localSum += __shfl_xor(localSum, 4);
		localSum += __shfl_xor(localSum, 2);
		localSum += __shfl_xor(localSum, 1);
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = localSum;
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

	int* dev_a, * d_accTemp, * d_accTemp2,*temp_dev_a;
	int h_accTemp = 0;
	int h_accTemp2 = 0;

	short* h_dev_a = (short*)malloc(size * sizeof(short));
	short* h_dev_a2 = (short*)malloc(size * sizeof(short));


	//cudaStatus = cudaDeviceSynchronize();
	
	cudaStatus = cudaMalloc((void**)&dev_a, size / 4 * sizeof(int));
	//cudaStatus = cudaMalloc((void**)&temp_dev_a, size / 2 * sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_accTemp, 1 * sizeof(int));
	
	cudaStatus = cudaMalloc((void**)&d_accTemp2, blocks * sizeof(int));
	int* h_odata = (int*)malloc(blocks * sizeof(int));
	int gpu_sum = 0;

	cudaStatus = cudaMemcpy(h_dev_a, a, size * sizeof(short), cudaMemcpyDeviceToHost);


	cudaStream_t stream1, stream2, stream3, stream4, stream5, stream6, stream7, stream8;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	cudaStreamCreate(&stream4);
	cudaStreamCreate(&stream5);
	cudaStreamCreate(&stream6);
	cudaStreamCreate(&stream7);
	cudaStreamCreate(&stream8);

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
		plusOneShort1 << <1536, 768, 0, stream1 >> > ((short*)a, size, skip);
		plusOneShort2 << <blocks, threads, 0, stream2 >> > ((short*)a, size, skip);
		plusOneShort3 << <blocks, threads, 0, stream3 >> > ((short*)a + size / 4, size, skip);
		plusOneShort4 << <blocks, threads, 0, stream4 >> > ((short*)a + size / 4, size, skip);
		plusOneShort5 << <blocks, threads, 0, stream5 >> > ((short*)a + size / 2, size, skip);
		plusOneShort6 << <blocks, threads, 0, stream6 >> > ((short*)a + size / 2, size, skip);
		plusOneShort7 << <blocks, threads, 0, stream7 >> > ((short*)a + 3 * size / 4, size, skip);
		plusOneShort8 << <blocks, threads, 0, stream8 >> > ((short*)a + 3 * size / 4, size, skip);
		cudaStatus = cudaDeviceSynchronize();
	}

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream3);
	cudaStreamDestroy(stream4);
	cudaStreamDestroy(stream5);
	cudaStreamDestroy(stream6);
	cudaStreamDestroy(stream7);
	cudaStreamDestroy(stream8);


	multiplyKernel << <blocks , threads>> > ((short*)a, (int*) dev_a, size);
	//cudaStatus = cudaMemcpy(h_dev_a2, dev_a, size/4  * sizeof(int), cudaMemcpyDeviceToHost);

	sumKernel << <blocks , threads >> > ((int*) dev_a, (int*) d_accTemp, size/4);

	reduceSmemUnrollShfl << <blocks, threads >> > ((int*) dev_a, (int*) d_accTemp2, size/4);
	cudaMemcpy(h_odata, d_accTemp2, blocks * sizeof(int), cudaMemcpyDeviceToHost);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "deviceReduceKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	
	cudaStatus = cudaMemcpy(&h_accTemp, d_accTemp, sizeof(int), cudaMemcpyDeviceToHost);


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
				fprintf(fptr, "%d\t%lld\t%lli\n", u32LoopCount, h_accTemp, h_accTemp2);
				fclose(fptr);
			}
		}
		else {
			if (AnalysisFile == 1) {
				fprintf(fptr, "%d\t%lld\t%lld\n", u32LoopCount, h_accTemp, h_accTemp2);
				fclose(fptr);
			}
		}
	}

	for (int i = 0; i < blocks; i++)
	{
		gpu_sum += h_odata[i];
	}

	printf("%d\t: %d \n", u32LoopCount,gpu_sum);
	fclose(fptr);
	// Don't forget to free allocated memory on both host and device
	if (CPUresult == 1) {
		free(h_dev_a);
		//free(h_dev_a2);
}
	free(h_odata);



	return cudaStatus;
}
