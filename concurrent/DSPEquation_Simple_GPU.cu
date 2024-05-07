
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
	for (int i = index; i < numElements / 4; i += stride)
	{
		a[i * 4] = (a[i * 4] - a[i * 4 + 2]);
		a[i * 4 + 1] = (a[i * 4 + 1] - a[i * 4 + 3]);
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

	int* dev_a, * d_accTemp, * d_accTemp2, * d_accTemp22, * d_accTemp222, *temp_dev_a, * dev_a2, * dev_a3,* dev_a4;
	int h_accTemp = 0;
	int h_accTemp2 = 0;

	short* h_dev_a = (short*)malloc(size * sizeof(short));
	short* h_dev_a2 = (short*)malloc(size * sizeof(short));


	//cudaStatus = cudaDeviceSynchronize();
	
	cudaStatus = cudaMalloc((void**)&dev_a, size / 4 * sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_a2, 100 * size / 4 * sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_a3, size / 4 * sizeof(int));
	cudaStatus = cudaMalloc((void**)&dev_a4, size / 4 * sizeof(int));
	//cudaStatus = cudaMalloc((void**)&temp_dev_a, size / 2 * sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_accTemp, 101 * sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_accTemp22, sizeof(int));
	cudaStatus = cudaMalloc((void**)&d_accTemp222, sizeof(int));
	
	cudaStatus = cudaMalloc((void**)&d_accTemp2, blocks * sizeof(int));
	int* h_odata = (int*)malloc(blocks * sizeof(int));
	int gpu_sum = 0;

	cudaStatus = cudaMemcpy(h_dev_a, a, size * sizeof(short), cudaMemcpyDeviceToHost);


	cudaStream_t stream1, stream2, stream3, stream4, stream5, stream6, stream7, stream8, stream9, stream10
		,stream11, stream12, stream13, stream14, stream15, stream16, stream17, stream18, stream19, stream20
		,stream21, stream22, stream23, stream24, stream25, stream26, stream27, stream28, stream29, stream30
		,stream31, stream32, stream33, stream34, stream35, stream36, stream37, stream38, stream39, stream40
		,stream41, stream42, stream43, stream44, stream45, stream46, stream47, stream48, stream49, stream50
		,stream51, stream52, stream53, stream54, stream55, stream56, stream57, stream58, stream59, stream60
		,stream61, stream62, stream63, stream64, stream65, stream66, stream67, stream68, stream69, stream70
		,stream71, stream72, stream73, stream74, stream75, stream76, stream77, stream78, stream79, stream80
		,stream81, stream82, stream83, stream84, stream85, stream86, stream87, stream88, stream89, stream90
		,stream91, stream92, stream93, stream94, stream95, stream96, stream97, stream98, stream99, stream100;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	cudaStreamCreate(&stream4);
	cudaStreamCreate(&stream5);
	cudaStreamCreate(&stream6);
	cudaStreamCreate(&stream7);
	cudaStreamCreate(&stream8);
	cudaStreamCreate(&stream9);
	cudaStreamCreate(&stream10);
	cudaStreamCreate(&stream11);
	cudaStreamCreate(&stream12);
	cudaStreamCreate(&stream13);
	cudaStreamCreate(&stream14);
	cudaStreamCreate(&stream15);
	cudaStreamCreate(&stream16);
	cudaStreamCreate(&stream17);
	cudaStreamCreate(&stream18);
	cudaStreamCreate(&stream19);
	cudaStreamCreate(&stream20);
	cudaStreamCreate(&stream21);
	cudaStreamCreate(&stream22);
	cudaStreamCreate(&stream23);
	cudaStreamCreate(&stream24);
	cudaStreamCreate(&stream25);
	cudaStreamCreate(&stream26);
	cudaStreamCreate(&stream27);
	cudaStreamCreate(&stream28);
	cudaStreamCreate(&stream29);
	cudaStreamCreate(&stream30);	
	cudaStreamCreate(&stream31);
	cudaStreamCreate(&stream32);
	cudaStreamCreate(&stream33);
	cudaStreamCreate(&stream34);
	cudaStreamCreate(&stream35);
	cudaStreamCreate(&stream36);
	cudaStreamCreate(&stream37);
	cudaStreamCreate(&stream38);
	cudaStreamCreate(&stream39);
	cudaStreamCreate(&stream40);	
	cudaStreamCreate(&stream41);
	cudaStreamCreate(&stream42);
	cudaStreamCreate(&stream43);
	cudaStreamCreate(&stream44);
	cudaStreamCreate(&stream45);
	cudaStreamCreate(&stream46);
	cudaStreamCreate(&stream47);
	cudaStreamCreate(&stream48);
	cudaStreamCreate(&stream49);
	cudaStreamCreate(&stream50);
	cudaStreamCreate(&stream51);
	cudaStreamCreate(&stream52);
	cudaStreamCreate(&stream53);
	cudaStreamCreate(&stream54);
	cudaStreamCreate(&stream55);
	cudaStreamCreate(&stream56);
	cudaStreamCreate(&stream57);
	cudaStreamCreate(&stream58);
	cudaStreamCreate(&stream59);
	cudaStreamCreate(&stream60);
	cudaStreamCreate(&stream61);
	cudaStreamCreate(&stream62);
	cudaStreamCreate(&stream63);
	cudaStreamCreate(&stream64);
	cudaStreamCreate(&stream65);
	cudaStreamCreate(&stream66);
	cudaStreamCreate(&stream67);
	cudaStreamCreate(&stream68);
	cudaStreamCreate(&stream69);
	cudaStreamCreate(&stream70);
	cudaStreamCreate(&stream71);
	cudaStreamCreate(&stream72);
	cudaStreamCreate(&stream73);
	cudaStreamCreate(&stream74);
	cudaStreamCreate(&stream75);
	cudaStreamCreate(&stream76);
	cudaStreamCreate(&stream77);
	cudaStreamCreate(&stream78);
	cudaStreamCreate(&stream79);
	cudaStreamCreate(&stream80);
	cudaStreamCreate(&stream81);
	cudaStreamCreate(&stream82);
	cudaStreamCreate(&stream83);
	cudaStreamCreate(&stream84);
	cudaStreamCreate(&stream85);
	cudaStreamCreate(&stream86);
	cudaStreamCreate(&stream87);
	cudaStreamCreate(&stream88);
	cudaStreamCreate(&stream89);
	cudaStreamCreate(&stream90);
	cudaStreamCreate(&stream91);
	cudaStreamCreate(&stream92);
	cudaStreamCreate(&stream93);
	cudaStreamCreate(&stream94);
	cudaStreamCreate(&stream95);
	cudaStreamCreate(&stream96);
	cudaStreamCreate(&stream97);
	cudaStreamCreate(&stream98);
	cudaStreamCreate(&stream99);
	cudaStreamCreate(&stream100);

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
	//cudaStatus = cudaMemcpy(h_dev_a2, dev_a, size/4  * sizeof(int), cudaMemcpyDeviceToHost);


	for (int ii = 0; ii < 100; ii++)
	{
		cudaMemcpy(dev_a2 + ii* size/4, dev_a, size/4 * sizeof(short), cudaMemcpyDeviceToHost);
	}


	sumKernel << <blocks, threads >> > ((int*)dev_a, (int*)d_accTemp, size / 4);
	//sumKernel << <blocks, threads >> > ((int*)dev_a2, (int*)d_accTemp2, 100*size / 4);
	sumKernel << <blocks, threads, 0, stream1 >> > ((int*)dev_a2 + 0 * size / 4, (int*)d_accTemp+1, size / 4);
	sumKernel << <blocks, threads, 0, stream2 >> > ((int*)dev_a3 , (int*)d_accTemp22, size / 4);
	sumKernel << <blocks, threads, 0, stream3 >> > ((int*)dev_a4 , (int*)d_accTemp222, size / 4);
	sumKernel << <blocks, threads, 0, stream4 >> > ((int*)dev_a2 + 3 * size / 4, (int*)d_accTemp + 4, size / 4);
	sumKernel << <blocks, threads, 0, stream5 >> > ((int*)dev_a2 + 4 * size / 4, (int*)d_accTemp + 5, size / 4);
	sumKernel << <blocks, threads, 0, stream6 >> > ((int*)dev_a2 + 5 * size / 4, (int*)d_accTemp + 6, size / 4);
	sumKernel << <blocks, threads, 0, stream7 >> > ((int*)dev_a2 + 6 * size / 4, (int*)d_accTemp + 7, size / 4);
	sumKernel << <blocks, threads, 0, stream8 >> > ((int*)dev_a2 + 7 * size / 4, (int*)d_accTemp + 8, size / 4);
	sumKernel << <blocks, threads, 0, stream9 >> > ((int*)dev_a2 + 8 * size / 4, (int*)d_accTemp + 9, size / 4);
	sumKernel << <blocks, threads, 0, stream10 >> > ((int*)dev_a2 + 9 * size / 4, (int*)d_accTemp+10 , size / 4);
	sumKernel << <blocks, threads, 0, stream11 >> > ((int*)dev_a2 + 10 * size / 4, (int*)d_accTemp + 11, size / 4);
	sumKernel << <blocks, threads, 0, stream12 >> > ((int*)dev_a2 + 11 * size / 4, (int*)d_accTemp + 12, size / 4);
	sumKernel << <blocks, threads, 0, stream13 >> > ((int*)dev_a2 + 12 * size / 4, (int*)d_accTemp + 13, size / 4);
	sumKernel << <blocks, threads, 0, stream14 >> > ((int*)dev_a2 + 13 * size / 4, (int*)d_accTemp + 14, size / 4);
	sumKernel << <blocks, threads, 0, stream15 >> > ((int*)dev_a2 + 14 * size / 4, (int*)d_accTemp + 15, size / 4);
	sumKernel << <blocks, threads, 0, stream16 >> > ((int*)dev_a2 + 15 * size / 4, (int*)d_accTemp + 16, size / 4);
	sumKernel << <blocks, threads, 0, stream17 >> > ((int*)dev_a2 + 16 * size / 4, (int*)d_accTemp + 17, size / 4);
	sumKernel << <blocks, threads, 0, stream18 >> > ((int*)dev_a2 + 17 * size / 4, (int*)d_accTemp + 18, size / 4);
	sumKernel << <blocks, threads, 0, stream19 >> > ((int*)dev_a2 + 18 * size / 4, (int*)d_accTemp + 19, size / 4);
	sumKernel << <blocks, threads, 0, stream20 >> > ((int*)dev_a2 + 19 * size / 4, (int*)d_accTemp + 20, size / 4);
	sumKernel << <blocks, threads, 0, stream21 >> > ((int*)dev_a2 + 20 * size / 4, (int*)d_accTemp + 21, size / 4);
	sumKernel << <blocks, threads, 0, stream22 >> > ((int*)dev_a2 + 21 * size / 4, (int*)d_accTemp + 22, size / 4);
	sumKernel << <blocks, threads, 0, stream23 >> > ((int*)dev_a2 + 22 * size / 4, (int*)d_accTemp + 23, size / 4);
	sumKernel << <blocks, threads, 0, stream24 >> > ((int*)dev_a2 + 23 * size / 4, (int*)d_accTemp + 24, size / 4);
	sumKernel << <blocks, threads, 0, stream25 >> > ((int*)dev_a2 + 24 * size / 4, (int*)d_accTemp + 25, size / 4);
	sumKernel << <blocks, threads, 0, stream26 >> > ((int*)dev_a2 + 25 * size / 4, (int*)d_accTemp + 26, size / 4);
	sumKernel << <blocks, threads, 0, stream27 >> > ((int*)dev_a2 + 26 * size / 4, (int*)d_accTemp + 27, size / 4);
	sumKernel << <blocks, threads, 0, stream28 >> > ((int*)dev_a2 + 27 * size / 4, (int*)d_accTemp + 28, size / 4);
	sumKernel << <blocks, threads, 0, stream29 >> > ((int*)dev_a2 + 28 * size / 4, (int*)d_accTemp + 29, size / 4);
	sumKernel << <blocks, threads, 0, stream30 >> > ((int*)dev_a2 + 29 * size / 4, (int*)d_accTemp + 30, size / 4);
	sumKernel << <blocks, threads, 0, stream31 >> > ((int*)dev_a2 + 30 * size / 4, (int*)d_accTemp+ 31, size / 4);
	sumKernel << <blocks, threads, 0, stream32 >> > ((int*)dev_a2 + 31 * size / 4, (int*)d_accTemp+32, size / 4);
	sumKernel << <blocks, threads, 0, stream33 >> > ((int*)dev_a2 + 32 * size / 4, (int*)d_accTemp+33, size / 4);
	sumKernel << <blocks, threads, 0, stream34 >> > ((int*)dev_a2 + 33 * size / 4, (int*)d_accTemp+34, size / 4);
	sumKernel << <blocks, threads, 0, stream35 >> > ((int*)dev_a2 + 34 * size / 4, (int*)d_accTemp+35, size / 4);
	sumKernel << <blocks, threads, 0, stream36 >> > ((int*)dev_a2 + 35 * size / 4, (int*)d_accTemp+36, size / 4);
	sumKernel << <blocks, threads, 0, stream37 >> > ((int*)dev_a2 + 36 * size / 4, (int*)d_accTemp+37, size / 4);
	sumKernel << <blocks, threads, 0, stream38 >> > ((int*)dev_a2 + 37 * size / 4, (int*)d_accTemp+38, size / 4);
	sumKernel << <blocks, threads, 0, stream39 >> > ((int*)dev_a2 + 38 * size / 4, (int*)d_accTemp + 39, size / 4);
	sumKernel << <blocks, threads, 0, stream40 >> > ((int*)dev_a2 + 39 * size / 4, (int*)d_accTemp + 40, size / 4);
	sumKernel << <blocks, threads, 0, stream41 >> > ((int*)dev_a2 + 40 * size / 4, (int*)d_accTemp + 41, size / 4);
	sumKernel << <blocks, threads, 0, stream42 >> > ((int*)dev_a2 + 41 * size / 4, (int*)d_accTemp + 42, size / 4);
	sumKernel << <blocks, threads, 0, stream43 >> > ((int*)dev_a2 + 42 * size / 4, (int*)d_accTemp + 43, size / 4);
	sumKernel << <blocks, threads, 0, stream44 >> > ((int*)dev_a2 + 43 * size / 4, (int*)d_accTemp + 44, size / 4);
	sumKernel << <blocks, threads, 0, stream45 >> > ((int*)dev_a2 + 44 * size / 4, (int*)d_accTemp + 45, size / 4);
	sumKernel << <blocks, threads, 0, stream46 >> > ((int*)dev_a2 + 45 * size / 4, (int*)d_accTemp + 46, size / 4);
	sumKernel << <blocks, threads, 0, stream47 >> > ((int*)dev_a2 + 46 * size / 4, (int*)d_accTemp + 47, size / 4);
	sumKernel << <blocks, threads, 0, stream48 >> > ((int*)dev_a2 + 47 * size / 4, (int*)d_accTemp + 48, size / 4);
	sumKernel << <blocks, threads, 0, stream49 >> > ((int*)dev_a2 + 48 * size / 4, (int*)d_accTemp + 49, size / 4);
	sumKernel << <blocks, threads, 0, stream50 >> > ((int*)dev_a2 + 49 * size / 4, (int*)d_accTemp + 50, size / 4);
	sumKernel << <blocks, threads, 0, stream51 >> > ((int*)dev_a2 + 50 * size / 4, (int*)d_accTemp + 51, size / 4);
	sumKernel << <blocks, threads, 0, stream52 >> > ((int*)dev_a2 + 51 * size / 4, (int*)d_accTemp + 52, size / 4);
	sumKernel << <blocks, threads, 0, stream53 >> > ((int*)dev_a2 + 52 * size / 4, (int*)d_accTemp + 53, size / 4);
	sumKernel << <blocks, threads, 0, stream54 >> > ((int*)dev_a2 + 53 * size / 4, (int*)d_accTemp + 54, size / 4);
	sumKernel << <blocks, threads, 0, stream55 >> > ((int*)dev_a2 + 54 * size / 4, (int*)d_accTemp + 55, size / 4);
	sumKernel << <blocks, threads, 0, stream56 >> > ((int*)dev_a2 + 55 * size / 4, (int*)d_accTemp + 56, size / 4);
	sumKernel << <blocks, threads, 0, stream57 >> > ((int*)dev_a2 + 56 * size / 4, (int*)d_accTemp + 57, size / 4);
	sumKernel << <blocks, threads, 0, stream58 >> > ((int*)dev_a2 + 57 * size / 4, (int*)d_accTemp + 58, size / 4);
	sumKernel << <blocks, threads, 0, stream59 >> > ((int*)dev_a2 + 58 * size / 4, (int*)d_accTemp + 59, size / 4);
	sumKernel << <blocks, threads, 0, stream60 >> > ((int*)dev_a2 + 59 * size / 4, (int*)d_accTemp + 60, size / 4);
	sumKernel << <blocks, threads, 0, stream61 >> > ((int*)dev_a2 + 60 * size / 4, (int*)d_accTemp + 61, size / 4);
	sumKernel << <blocks, threads, 0, stream62 >> > ((int*)dev_a2 + 61 * size / 4, (int*)d_accTemp + 62, size / 4);
	sumKernel << <blocks, threads, 0, stream63 >> > ((int*)dev_a2 + 62 * size / 4, (int*)d_accTemp + 63, size / 4);
	sumKernel << <blocks, threads, 0, stream64 >> > ((int*)dev_a2 + 63 * size / 4, (int*)d_accTemp + 64, size / 4);
	sumKernel << <blocks, threads, 0, stream65 >> > ((int*)dev_a2 + 64 * size / 4, (int*)d_accTemp + 65, size / 4);
	sumKernel << <blocks, threads, 0, stream66 >> > ((int*)dev_a2 + 65 * size / 4, (int*)d_accTemp + 66, size / 4);
	sumKernel << <blocks, threads, 0, stream67 >> > ((int*)dev_a2 + 66 * size / 4, (int*)d_accTemp + 67, size / 4);
	sumKernel << <blocks, threads, 0, stream68 >> > ((int*)dev_a2 + 67 * size / 4, (int*)d_accTemp + 68, size / 4);
	sumKernel << <blocks, threads, 0, stream69 >> > ((int*)dev_a2 + 68 * size / 4, (int*)d_accTemp + 69, size / 4);
	sumKernel << <blocks, threads, 0, stream70 >> > ((int*)dev_a2 + 69 * size / 4, (int*)d_accTemp + 70, size / 4);
	sumKernel << <blocks, threads, 0, stream71 >> > ((int*)dev_a2 + 70 * size / 4, (int*)d_accTemp + 71, size / 4);
	sumKernel << <blocks, threads, 0, stream72 >> > ((int*)dev_a2 + 71 * size / 4, (int*)d_accTemp + 72, size / 4);
	sumKernel << <blocks, threads, 0, stream73 >> > ((int*)dev_a2 + 72 * size / 4, (int*)d_accTemp + 73, size / 4);
	sumKernel << <blocks, threads, 0, stream74 >> > ((int*)dev_a2 + 73 * size / 4, (int*)d_accTemp + 74, size / 4);
	sumKernel << <blocks, threads, 0, stream75 >> > ((int*)dev_a2 + 74 * size / 4, (int*)d_accTemp + 75, size / 4);
	sumKernel << <blocks, threads, 0, stream76 >> > ((int*)dev_a2 + 75 * size / 4, (int*)d_accTemp + 76, size / 4);
	sumKernel << <blocks, threads, 0, stream77 >> > ((int*)dev_a2 + 76 * size / 4, (int*)d_accTemp + 77, size / 4);
	sumKernel << <blocks, threads, 0, stream78 >> > ((int*)dev_a2 + 77 * size / 4, (int*)d_accTemp + 78, size / 4);
	sumKernel << <blocks, threads, 0, stream79 >> > ((int*)dev_a2 + 78 * size / 4, (int*)d_accTemp + 79, size / 4);
	sumKernel << <blocks, threads, 0, stream80 >> > ((int*)dev_a2 + 79 * size / 4, (int*)d_accTemp + 80, size / 4);
	sumKernel << <blocks, threads, 0, stream81 >> > ((int*)dev_a2 + 80 * size / 4, (int*)d_accTemp + 81, size / 4);
	sumKernel << <blocks, threads, 0, stream82 >> > ((int*)dev_a2 + 81 * size / 4, (int*)d_accTemp + 82, size / 4);
	sumKernel << <blocks, threads, 0, stream83 >> > ((int*)dev_a2 + 82 * size / 4, (int*)d_accTemp + 83, size / 4);
	sumKernel << <blocks, threads, 0, stream84 >> > ((int*)dev_a2 + 83 * size / 4, (int*)d_accTemp + 84, size / 4);
	sumKernel << <blocks, threads, 0, stream85 >> > ((int*)dev_a2 + 84 * size / 4, (int*)d_accTemp + 85, size / 4);
	sumKernel << <blocks, threads, 0, stream86 >> > ((int*)dev_a2 + 85 * size / 4, (int*)d_accTemp + 86, size / 4);
	sumKernel << <blocks, threads, 0, stream87 >> > ((int*)dev_a2 + 86 * size / 4, (int*)d_accTemp + 87, size / 4);
	sumKernel << <blocks, threads, 0, stream88 >> > ((int*)dev_a2 + 87 * size / 4, (int*)d_accTemp + 88, size / 4);
	sumKernel << <blocks, threads, 0, stream89 >> > ((int*)dev_a2 + 88 * size / 4, (int*)d_accTemp + 89, size / 4);
	sumKernel << <blocks, threads, 0, stream90 >> > ((int*)dev_a2 + 89 * size / 4, (int*)d_accTemp + 90, size / 4);
	sumKernel << <blocks, threads, 0, stream91 >> > ((int*)dev_a2 + 90 * size / 4, (int*)d_accTemp + 91, size / 4);
	sumKernel << <blocks, threads, 0, stream92 >> > ((int*)dev_a2 + 91 * size / 4, (int*)d_accTemp + 92, size / 4);
	sumKernel << <blocks, threads, 0, stream93 >> > ((int*)dev_a2 + 92 * size / 4, (int*)d_accTemp + 93, size / 4);
	sumKernel << <blocks, threads, 0, stream94 >> > ((int*)dev_a2 + 93 * size / 4, (int*)d_accTemp + 94, size / 4);
	sumKernel << <blocks, threads, 0, stream95 >> > ((int*)dev_a2 + 94 * size / 4, (int*)d_accTemp + 95, size / 4);
	sumKernel << <blocks, threads, 0, stream96 >> > ((int*)dev_a2 + 95 * size / 4, (int*)d_accTemp + 96, size / 4);
	sumKernel << <blocks, threads, 0, stream97 >> > ((int*)dev_a2 + 96 * size / 4, (int*)d_accTemp + 97, size / 4);
	sumKernel << <blocks, threads, 0, stream98 >> > ((int*)dev_a2 + 97 * size / 4, (int*)d_accTemp + 98, size / 4);
	sumKernel << <blocks, threads, 0, stream99 >> > ((int*)dev_a2 + 98 * size / 4, (int*)d_accTemp + 99, size / 4);
	sumKernel << <blocks, threads, 0, stream100 >> > ((int*)dev_a2 + 99 * size / 4, (int*)d_accTemp+100, size / 4);



	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
	cudaStreamDestroy(stream3);
	cudaStreamDestroy(stream4);
	cudaStreamDestroy(stream5);
	cudaStreamDestroy(stream6);
	cudaStreamDestroy(stream7);
	cudaStreamDestroy(stream8);
	cudaStreamDestroy(stream9);
	cudaStreamDestroy(stream10);
	cudaStreamDestroy(stream11);
	cudaStreamDestroy(stream12);
	cudaStreamDestroy(stream13);
	cudaStreamDestroy(stream14);
	cudaStreamDestroy(stream15);
	cudaStreamDestroy(stream16);
	cudaStreamDestroy(stream17);
	cudaStreamDestroy(stream18);
	cudaStreamDestroy(stream19);
	cudaStreamDestroy(stream20);
	cudaStreamDestroy(stream21);
	cudaStreamDestroy(stream22);
	cudaStreamDestroy(stream23);
	cudaStreamDestroy(stream24);
	cudaStreamDestroy(stream25);
	cudaStreamDestroy(stream26);
	cudaStreamDestroy(stream27);
	cudaStreamDestroy(stream28);
	cudaStreamDestroy(stream29);
	cudaStreamDestroy(stream30);
	cudaStreamDestroy(stream31);
	cudaStreamDestroy(stream32);
	cudaStreamDestroy(stream33);
	cudaStreamDestroy(stream34);
	cudaStreamDestroy(stream35);
	cudaStreamDestroy(stream36);
	cudaStreamDestroy(stream37);
	cudaStreamDestroy(stream38);
	cudaStreamDestroy(stream39);
	cudaStreamDestroy(stream40);
	cudaStreamDestroy(stream41);
	cudaStreamDestroy(stream42);
	cudaStreamDestroy(stream43);
	cudaStreamDestroy(stream44);
	cudaStreamDestroy(stream45);
	cudaStreamDestroy(stream46);
	cudaStreamDestroy(stream47);
	cudaStreamDestroy(stream48);
	cudaStreamDestroy(stream49);
	cudaStreamDestroy(stream50);
	cudaStreamDestroy(stream51);
	cudaStreamDestroy(stream52);
	cudaStreamDestroy(stream53);
	cudaStreamDestroy(stream54);
	cudaStreamDestroy(stream55);
	cudaStreamDestroy(stream56);
	cudaStreamDestroy(stream57);
	cudaStreamDestroy(stream58);
	cudaStreamDestroy(stream59);
	cudaStreamDestroy(stream60);
	cudaStreamDestroy(stream61);
	cudaStreamDestroy(stream62);
	cudaStreamDestroy(stream63);
	cudaStreamDestroy(stream64);
	cudaStreamDestroy(stream65);
	cudaStreamDestroy(stream66);
	cudaStreamDestroy(stream67);
	cudaStreamDestroy(stream68);
	cudaStreamDestroy(stream69);
	cudaStreamDestroy(stream70);
	cudaStreamDestroy(stream71);
	cudaStreamDestroy(stream72);
	cudaStreamDestroy(stream73);
	cudaStreamDestroy(stream74);
	cudaStreamDestroy(stream75);
	cudaStreamDestroy(stream76);
	cudaStreamDestroy(stream77);
	cudaStreamDestroy(stream78);
	cudaStreamDestroy(stream79);
	cudaStreamDestroy(stream80);
	cudaStreamDestroy(stream81);
	cudaStreamDestroy(stream82);
	cudaStreamDestroy(stream83);
	cudaStreamDestroy(stream84);
	cudaStreamDestroy(stream85);
	cudaStreamDestroy(stream86);
	cudaStreamDestroy(stream87);
	cudaStreamDestroy(stream88);
	cudaStreamDestroy(stream89);
	cudaStreamDestroy(stream90);
	cudaStreamDestroy(stream91);
	cudaStreamDestroy(stream92);
	cudaStreamDestroy(stream93);
	cudaStreamDestroy(stream94);
	cudaStreamDestroy(stream95);
	cudaStreamDestroy(stream96);
	cudaStreamDestroy(stream97);
	cudaStreamDestroy(stream98);
	cudaStreamDestroy(stream99);
	cudaStreamDestroy(stream100);

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
	cudaFree(dev_a2);
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
