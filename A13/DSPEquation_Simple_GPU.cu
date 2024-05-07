
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



__global__ void plusOneShort1(short* a, __int64 numElements, unsigned long skip, int* out)
{
	__shared__ int smem[SMEMDIM];
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
#pragma unroll
	for (int i = index; i < numElements / 4; i += stride)
	{
		smem[0] = a[i * 48];
		smem[1] = a[i * 48 + 1];
		smem[2] = a[i * 48 + 2];
		smem[3] = a[i * 48 + 3];
		smem[4] = a[i * 48 + 4];
		smem[5] = a[i * 48 + 5];
		smem[6] = a[i * 48 + 6];
		smem[7] = a[i * 48 + 7];
		smem[8] = a[i * 48 + 8];
		smem[9] = a[i * 48 + 9];
		smem[10] = a[i * 48 + 10];
		smem[11] = a[i * 48 + 11];
		smem[12] = a[i * 48 + 12];
		smem[13] = a[i * 48 + 13];
		smem[14] = a[i * 48 + 14];
		smem[15] = a[i * 48 + 15];
		smem[16] = a[i * 48 + 16];
		smem[17] = a[i * 48 + 17];
		smem[18] = a[i * 48 + 18];
		smem[19] = a[i * 48 + 19];
		smem[20] = a[i * 48 + 20];
		smem[21] = a[i * 48 + 21];
		smem[22] = a[i * 48 + 22];
		smem[23] = a[i * 48 + 23];
		smem[24] = a[i * 48 + 24];
		smem[25] = a[i * 48 + 25];
		smem[26] = a[i * 48 + 26];
		smem[27] = a[i * 48 + 27];
		smem[28] = a[i * 48 + 28];
		smem[29] = a[i * 48 + 29];
		smem[30] = a[i * 48 + 30];
		smem[31] = a[i * 48 + 31];
		smem[32] = a[i * 48 + 32];
		smem[33] = a[i * 48 + 33];
		smem[34] = a[i * 48 + 34];
		smem[35] = a[i * 48 + 35];
		smem[36] = a[i * 48 + 36];
		smem[37] = a[i * 48 + 37];
		smem[38] = a[i * 48 + 38];
		smem[39] = a[i * 48 + 39];
		smem[40] = a[i * 48 + 40];
		smem[41] = a[i * 48 + 41];
		smem[42] = a[i * 48 + 42];
		smem[43] = a[i * 48 + 43];
		smem[44] = a[i * 48 + 44];
		smem[45] = a[i * 48 + 45];
		smem[46] = a[i * 48 + 46];
		smem[47] = a[i * 48 + 47];
		smem[48] = smem[24] - smem[0];
		smem[49] = smem[26] - smem[2];
		smem[50] = smem[28] - smem[4];
		smem[51] = smem[6] - smem[30];
		smem[52] = smem[8] - smem[32];
		smem[53] = smem[10] - smem[34];
		smem[54] = smem[12] - smem[36];
		smem[55] = smem[14] - smem[38];
		smem[56] = smem[16] - smem[40];
		smem[57] = smem[42] - smem[18];
		smem[58] = smem[44] - smem[20];
		smem[59] = smem[46] - smem[22];
		smem[60] = smem[1] - smem[25];
		smem[61] = smem[3] - smem[27];
		smem[62] = smem[5] - smem[29];
		smem[63] = smem[7] - smem[31];
		smem[64] = smem[9] - smem[33];
		smem[65] = smem[11] - smem[35];
		smem[66] = smem[13] - smem[37];
		smem[67] = smem[15] - smem[39];
		smem[68] = smem[17] - smem[41];
		smem[69] = smem[19] - smem[43];
		smem[70] = smem[21] - smem[45];
		smem[71] = smem[23] - smem[47];
		int tempResult = 0;
		tempResult = smem[48] * smem[60] + smem[49] * smem[61] + smem[50] * smem[62] + smem[51] * smem[63] + smem[52] * smem[64] + smem[53] * smem[65] + smem[54] * smem[66] + smem[55] * smem[67] + smem[56] * smem[68] + smem[57] * smem[69] + smem[58] * smem[70] + smem[59] * smem[71];
		out[i] = tempResult;
	}

}

__global__ void plusOneShort2(short* a, __int64 numElements, int* out)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
#pragma unroll
	for (int i = index; i < numElements / 48; i += stride)
	{
		int a1 = a[i * 48];
		int a2 = a[i * 48 + 1];
		int a3 = a[i * 48 + 2];
		int a4 = a[i * 48 + 3];
		int a5 = a[i * 48 + 4];
		int a6 = a[i * 48 + 5];
		int a7 = a[i * 48 + 6];
		int a8 = a[i * 48 + 7];
		int a9 = a[i * 48 + 8];
		int a10 = a[i * 48 + 9];
		int a11 = a[i * 48 + 10];
		int a12 = a[i * 48 + 11];
		int a13 = a[i * 48 + 12];
		int a14 = a[i * 48 + 13];
		int a15 = a[i * 48 + 14];
		int a16 = a[i * 48 + 15];
		int a17 = a[i * 48 + 16];
		int a18 = a[i * 48 + 17];
		int a19 = a[i * 48 + 18];
		int a20 = a[i * 48 + 19];
		int a21 = a[i * 48 + 20];
		int a22 = a[i * 48 + 21];
		int a23 = a[i * 48 + 22];
		int a24 = a[i * 48 + 23];
		int a25 = a[i * 48 + 24];
		int a26 = a[i * 48 + 25];
		int a27 = a[i * 48 + 26];
		int a28 = a[i * 48 + 27];
		int a29 = a[i * 48 + 28];
		int a30 = a[i * 48 + 29];
		int a31 = a[i * 48 + 30];
		int a32 = a[i * 48 + 31];
		int a33 = a[i * 48 + 32];
		int a34 = a[i * 48 + 33];
		int a35 = a[i * 48 + 34];
		int a36 = a[i * 48 + 35];
		int a37 = a[i * 48 + 36];
		int a38 = a[i * 48 + 37];
		int a39 = a[i * 48 + 38];
		int a40 = a[i * 48 + 39];
		int a41 = a[i * 48 + 40];
		int a42 = a[i * 48 + 41];
		int a43 = a[i * 48 + 42];
		int a44 = a[i * 48 + 43];
		int a45 = a[i * 48 + 44];
		int a46 = a[i * 48 + 45];
		int a47 = a[i * 48 + 46];
		int a48 = a[i * 48 + 47];
		int temp = 0;
		temp = (a25 - a1) * (a2 - a26) + (a27 - a3) * (a4 - a28) + (a29 - a5) * (a6 - a30) + (a7 - a31) * (a8 - a32) + (a9 - a33) * (a10 - a34) + (a11 - a35) * (a12 - a36) + (a13 - a37) * (a14 - a38) + (a15 - a39) * (a16 - a40) + (a17 - a41) * (a18 - a42) + (a43 - a19) * (a20 - a44) + (a45 - a21) * (a22 - a46) + (a47 - a23) * (a24 - a48);
		out[i] = temp;
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

__inline__ __device__ int warpReduce(int mySum) {
	mySum += __shfl_xor(mySum, 16);
	mySum += __shfl_xor(mySum, 8);
	mySum += __shfl_xor(mySum, 4);
	mySum += __shfl_xor(mySum, 2);
	mySum += __shfl_xor(mySum, 1);
	return mySum;
}

__global__ void subtraction(int* a, int* out, unsigned int n) {
	__shared__ int smem[SMEMDIM];
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
#pragma unroll
	for (int i = idx;  i < n/4; i += stride)
	{
		int a1 = a[i*4];
		int a2 = a[i*4 + 1];
		int a3 = a[i*4+2];	
		int a4 = a[i*4 + 3];
		int temp1;
		int temp2;
		temp1 = a1 - a2;
		temp2 = a3 - a4;
		out[i] = temp1*temp2;
	}
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
__global__ void initializeArray(int* array) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		array[idx] = 1;
	}
}

__global__ void resetInteger(int* value) {
	*value = 0; // Reset integer value
}

// Helper function for using CUDA.

extern "C" cudaError_t GPU_Equation_PlusOne(void* a, unsigned long skip, unsigned long sample_size, __int64 size, int blocks, int threads, int u32LoopCount, int* h_odata, short* h_dev_a, short* h_dev_a2, int * dev_a, int* d_accTemp, int * d_accTemp2)
{
	cudaError_t cudaStatus = cudaSuccess;

	blocks = 48 * 32;
	threads = 768;
	clock_t start_Time, current_time;
	double elapsed_time;

	int CPUresult = 0;
	int CheckRaw = 0;
	int AnalysisFile = 1;

	int* check_dev =  (int*)malloc(size/48 * sizeof(int));
	int h_accTemp2 = 0;

	FILE* fptr;
	if (AnalysisFile == 1) {
		fptr = fopen("Analysis.txt", "a");
	}
	start_Time = clock();


	cudaStatus = cudaMemcpy(h_dev_a, a, size * sizeof(short), cudaMemcpyDeviceToHost);

	plusOneShort2 << <blocks, threads >> > ((short*)a, size, dev_a);
	cudaStatus = cudaMemcpy(check_dev, dev_a, size/48 * sizeof(int), cudaMemcpyDeviceToHost);
	reduceShfl << <blocks, threads >> > (dev_a, d_accTemp2, size / 48);
	cudaMemcpy(h_odata, d_accTemp2, 1 * sizeof(int), cudaMemcpyDeviceToHost);


	resetInteger << <1, 1 >> > ((int*)d_accTemp2);

	cudaStatus = cudaMemcpy(h_dev_a2, a, size * sizeof(short), cudaMemcpyDeviceToHost);
	cudaStatus = cudaDeviceSynchronize();
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.


	if (CPUresult == 1) {
		for (int i = 0; i < size/48; i++) {
					int a1 = h_dev_a[i * 48];
					int a2 = h_dev_a[i * 48 + 1];
					int a3 = h_dev_a[i * 48 + 2];
					int a4 = h_dev_a[i * 48 + 3];
					int a5 = h_dev_a[i * 48 + 4];
					int a6 = h_dev_a[i * 48 + 5];
					int a7 = h_dev_a[i * 48 + 6];
					int a8 = h_dev_a[i * 48 + 7];
					int a9 = h_dev_a[i * 48 + 8];
					int a10 = h_dev_a[i * 48 + 9];
					int a11 = h_dev_a[i * 48 + 10];
					int a12 = h_dev_a[i * 48 + 11];
					int a13 = h_dev_a[i * 48 + 12];
					int a14 = h_dev_a[i * 48 + 13];
					int a15 = h_dev_a[i * 48 + 14];
					int a16 = h_dev_a[i * 48 + 15];
					int a17 = h_dev_a[i * 48 + 16];
					int a18 = h_dev_a[i * 48 + 17];
					int a19 = h_dev_a[i * 48 + 18];
					int a20 = h_dev_a[i * 48 + 19];
					int a21 = h_dev_a[i * 48 + 20];
					int a22 = h_dev_a[i * 48 + 21];
					int a23 = h_dev_a[i * 48 + 22];
					int a24 = h_dev_a[i * 48 + 23];
					int a25 = h_dev_a[i * 48 + 24];
					int a26 = h_dev_a[i * 48 + 25];
					int a27 = h_dev_a[i * 48 + 26];
					int a28 = h_dev_a[i * 48 + 27];
					int a29 = h_dev_a[i * 48 + 28];
					int a30 = h_dev_a[i * 48 + 29];
					int a31 = h_dev_a[i * 48 + 30];
					int a32 = h_dev_a[i * 48 + 31];
					int a33 = h_dev_a[i * 48 + 32];
					int a34 = h_dev_a[i * 48 + 33];
					int a35 = h_dev_a[i * 48 + 34];
					int a36 = h_dev_a[i * 48 + 35];
					int a37 = h_dev_a[i * 48 + 36];
					int a38 = h_dev_a[i * 48 + 37];
					int a39 = h_dev_a[i * 48 + 38];
					int a40 = h_dev_a[i * 48 + 39];
					int a41 = h_dev_a[i * 48 + 40];
					int a42 = h_dev_a[i * 48 + 41];
					int a43 = h_dev_a[i * 48 + 42];
					int a44 = h_dev_a[i * 48 + 43];
					int a45 = h_dev_a[i * 48 + 44];
					int a46 = h_dev_a[i * 48 + 45];
					int a47 = h_dev_a[i * 48 + 46];
					int a48 = h_dev_a[i * 48 + 47];
					int temp = 0;
					temp = (a25 - a1) * (a2 - a26) + (a27 - a3) * (a4 - a28) + (a29 - a5) * (a6 - a30) + (a7 - a31) * (a8 - a32) + (a9 - a33) * (a10 - a34) + (a11 - a35) * (a12 - a36) + (a13 - a37) * (a14 - a38) + (a15 - a39) * (a16 - a40) + (a17 - a41) * (a18 - a42) + (a43 - a19) * (a20 - a44) + (a45 - a21) * (a22 - a46) + (a47 - a23) * (a24 - a48);
					h_accTemp2 += temp;
					//if(temp!=check_dev[i]) printf("\n%d\nCPU: %d\nGPU: %d\n", i, temp, check_dev[i]);
			
		}
	}


	if (CheckRaw != 1) {
		if (CPUresult == 1) {
			if (AnalysisFile == 1) {
				fprintf(fptr, "%d\t%d\t%d\n", u32LoopCount,h_accTemp2, h_odata[0]);
				fclose(fptr);
			}
		}
		else {
			if (AnalysisFile == 1) {
				fprintf(fptr, "%d\t%d\n", u32LoopCount, h_odata[0]);
				fclose(fptr);
			}
		}
	}


	fclose(fptr);
	// Get the current time
	current_time = clock();
	elapsed_time = ((double)(current_time - start_Time)) / CLOCKS_PER_SEC * 1000;
	printf("Elapsed Time: %.2f ms\r", elapsed_time);

	return cudaStatus;
}
