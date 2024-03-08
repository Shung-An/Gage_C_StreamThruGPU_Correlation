
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Windows.h>
#include <stdlib.h>
#include <time.h>


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



__global__ void plusOneShort(short *a, __int64 numElements, unsigned long skip)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i =index;  i%4==0 && i < numElements - 2; i+= stride)
	{
		a[i] = (a[i] - a[i + 2]);
		a[i+1] = (a[i+1] - a[i + 3]);
		a[i + 2] = 0;
		a[i + 3] = 0;
	}
}

__global__ void multiplyKernel(short* a, unsigned long long* dev_a,  __int64 numElements)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < numElements /4; i+=stride)
	{
		dev_a[i] = (unsigned long long)a[i * 4 ] * (unsigned long long)a[i * 4 + 1];
	}

}
__global__ void sumKernel(unsigned long long* dev_a, unsigned long long* d_accTemp, __int64 numElements)
{

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < numElements / 4; i+=stride) 
	{
		atomicAdd((unsigned long long *) d_accTemp, (unsigned long long)dev_a[i]);
	}
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
	time_t t = time(NULL);
	struct tm* tm = localtime(&t);
	char s[64];
	size_t ret = strftime(s, sizeof(s), "%c", tm);

	int CPUresult =0 ;
	int CheckRaw =0;
	int AnalysisFile = 1;

	FILE* fptr;
	if (AnalysisFile == 1) {
		fptr = fopen("Analysis.txt", "a");
	}

	// Launch a kernel on the GPU with one thread for each element.

	unsigned long long* dev_a, * d_accTemp;
	unsigned long long h_accTemp = 0;
	unsigned long long h_accTemp2 = 0;

	short* h_dev_a = (short*)malloc(size * sizeof(short));
	//unsigned long long* h_dev_a2 = (unsigned long long*)malloc(size * sizeof(unsigned long long));

	cudaStatus = cudaMemcpy(h_dev_a, a, size * sizeof(short), cudaMemcpyDeviceToHost);
	cudaStatus = cudaDeviceSynchronize();


		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy for h_dev_a failed: %s\n", cudaGetErrorString(cudaStatus));
			free(h_dev_a);
			cudaFree(a);
			return cudaStatus;
		}
	
	cudaStatus = cudaMalloc((void**)&dev_a, size / 4 * sizeof(unsigned long long));
	cudaStatus = cudaMalloc((void**)&d_accTemp, 1 * sizeof(unsigned long long));


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
		plusOneShort << <blocks, threads >> > ((short*)a, size, skip);

	}


	cudaStatus = cudaGetLastError();


	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	multiplyKernel << <blocks , threads>> > ((short*)a, (unsigned long long*) dev_a, size);
	//cudaStatus = cudaMemcpy(h_dev_a2, dev_a, size/4  * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "multiplyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}


	sumKernel << <blocks , threads >> > ((unsigned long long*) dev_a, (unsigned long long*) d_accTemp, size);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "sumKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}


	cudaStatus = cudaMemcpy(&h_accTemp, d_accTemp, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
	cudaFree(dev_a);
	cudaFree(d_accTemp);

	cudaStatus = cudaDeviceSynchronize();
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.


	if (CPUresult == 1) {
		for (int i = 0; i < size; i++) {
			if (i % 4 == 0 && i < size - 2)
			{
				//// Write to local disk
				if (1 == CheckRaw) {
					h_dev_a[i] = h_dev_a[i] - h_dev_a[i + 2];
					h_dev_a[i + 1] = h_dev_a[i + 1] - h_dev_a[i + 3];
					h_dev_a[i + 2] = 0;
					h_dev_a[i + 3] = 0;

					h_dev_a[i] = h_dev_a[i] * h_dev_a[i + 1];
					//printf("%d\t%d\t%d\n", i, h_dev_a[i], h_dev_a2[i / 4]);
					h_dev_a[i + 1] = 0;

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
	//if (CPUresult == 1) {
	//	for (int i = 0; i < size/4; i++) {
	//		if (h_dev_a[i * 4] != h_dev_a2[i])
	//		{
	//			fprintf(fptr, "%d\t %d\t % d\n",i, h_dev_a[i * 4], h_dev_a2[i]);
	//		}
	//	}
	//}

	if (CheckRaw != 1) {
		if (CPUresult == 1) {
			if (AnalysisFile == 1) {
				fprintf(fptr, "%d\t%lld\t%lld\n", u32LoopCount, h_accTemp, h_accTemp2);
				fclose(fptr);
			}
		}
		else {

			if (AnalysisFile == 1) {
				fprintf(fptr, "%d\t%lld\n", u32LoopCount, h_accTemp);
				fclose(fptr);
			}
		}
	}
	
	// Don't forget to free allocated memory on both host and device
	if (CPUresult == 1) {
		free(h_dev_a);
		//free(h_dev_a2);
}
	
	if (cudaStatus != cudaSuccess) 
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		return cudaStatus;
	}
	return cudaStatus;
}
