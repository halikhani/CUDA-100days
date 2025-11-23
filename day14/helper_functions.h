#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <random>

void printMatrix(const char* name, const float* matrix, int rows, int cols);
void initializeRandomMatrix(float* matrix, int size, float min_val, float max_val);
cudaError_t allocateDeviceMemory(float** device_pointer, size_t size); // wrapper for cudaMalloc
void freeDeviceMemory(float* device_pointer); // wrapper for cudaFree
void checkCudaStatus(cudaError_t status);
void checkCublasStatus(cublasStatus_t status);