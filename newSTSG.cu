#include "Filter.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include<iostream>
using namespace std;

void newSTSG(int win,float sampcorr, int m2,int  d2,int t, int s, int n_Years, int n_X, int PerStep, int n_B, int device, int StaOrEnd,int n_Y_d_vector_out,float* vector_in, float* vector_out/*,double N*/ )
{
	cudaSetDevice(device);

	int blkwidth = 16;
	int blkheight = 16;
	dim3 blocks(blkwidth, blkheight);
	dim3 grids(n_X % blkwidth == 0 ? n_X / blkwidth : n_X / blkwidth + 1, PerStep % blkheight == 0 ? PerStep / blkheight : PerStep / blkheight + 1);

	float* d_vector_in;
	size_t nBytes = n_X * PerStep * n_B * n_Years * sizeof(float);
	cudaMalloc((void**)&d_vector_in, nBytes);
	cudaMemcpy((void*)d_vector_in, (void*)vector_in, nBytes, cudaMemcpyHostToDevice);
	float* d_img_QA;
	nBytes = n_X * PerStep * n_B * n_Years * sizeof(float);
	cudaMalloc((void**)&d_img_QA, nBytes);
	cudaMemset((void*)d_img_QA, 0, nBytes);
	float* d_VI_Reference;
	nBytes = n_X * PerStep * n_B * sizeof(float);
	cudaMalloc((void**)&d_VI_Reference, nBytes);
	cudaMemset((void*)d_VI_Reference, 0, nBytes);
	float* d_distance_similarity;
	nBytes = n_X * PerStep * (n_B + 2) * n_Years * sizeof(float);
	cudaMalloc((void**)&d_distance_similarity, nBytes);
	cudaMemset((void*)d_distance_similarity, 0, nBytes);
	int* d_count_vec_res1;
	nBytes = n_X * PerStep * n_B * sizeof(int);
	cudaMalloc((void**)&d_count_vec_res1, nBytes);
	cudaMemset((void*)d_count_vec_res1, 0, nBytes);
	cudaDeviceSynchronize();

	Generate_VI_reference << <grids, blocks >> > ( d_vector_in, n_X, PerStep, n_B, n_Years, d_img_QA, d_VI_Reference, d_distance_similarity, d_count_vec_res1/*,N*/);
	cudaDeviceSynchronize();

	cudaFree((void*)d_distance_similarity);
	cudaFree((void*)d_count_vec_res1);

	float* d_res;
	nBytes = n_X * PerStep * (2 * win + 1) * (2 * win + 1) * 3 * sizeof(float);
	cudaMalloc((void**)&d_res, nBytes);
	cudaMemset((void*)d_res, 0, nBytes);
	cudaDeviceSynchronize();

	Compute_similar_pixel << <grids, blocks >> > (d_vector_in, d_img_QA, d_VI_Reference, n_X, PerStep, n_B, n_Years, win, d_res);
	cudaDeviceSynchronize();

	float* d_vector_out;
	nBytes = n_X * n_Y_d_vector_out * n_B * n_Years * sizeof(float);
	cudaMalloc((void**)&d_vector_out, nBytes);
	cudaMemset((void*)d_vector_out, 0, nBytes);
	int* d_index;
	nBytes = n_X * PerStep * (2 * win + 1) * (2 * win + 1) * sizeof(int);
	cudaMalloc((void**)&d_index, nBytes);
	cudaMemset((void*)d_index, 0, nBytes);
	int* d_index_1;
	nBytes = n_X * PerStep * (2 * win + 1) * (2 * win + 1) * 2 * sizeof(int);
	cudaMalloc((void**)&d_index_1, nBytes);
	cudaMemset((void*)d_index_1, 0, nBytes);
	float* d_res_1;
	nBytes = n_X * PerStep * (2 * win + 1) * (2 * win + 1) * 4 * sizeof(float);
	cudaMalloc((void**)&d_res_1, nBytes);
	cudaMemset((void*)d_res_1, 0, nBytes);
	float* d_res_2;
	nBytes = n_X * PerStep * n_B *4 * sizeof(float);
	cudaMalloc((void**)&d_res_2, nBytes);
	cudaMemset((void*)d_res_2, 0, nBytes);
	int* d_index_2;
	nBytes = n_X * PerStep * n_B * sizeof(int);
	cudaMalloc((void**)&d_index_2, nBytes);
	cudaMemset((void*)d_index_2, 0, nBytes);
	cudaDeviceSynchronize();

	STSG_filter << <grids, blocks >> > (d_vector_in, d_img_QA, d_VI_Reference, d_res, n_X, PerStep, n_B, n_Years, win, sampcorr, StaOrEnd, d_vector_out, d_index, d_index_1, d_res_1, d_res_2, d_index_2, m2, d2, t, s);
	cudaDeviceSynchronize();

	nBytes = n_X * n_Y_d_vector_out * n_B * n_Years * sizeof(float);
	cudaMemcpy((void*)vector_out, (void*)d_vector_out, nBytes, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaFree((void*)d_vector_in);
	cudaFree((void*)d_img_QA);
	cudaFree((void*)d_VI_Reference);
	cudaFree((void*)d_res);
	cudaFree((void*)d_vector_out);
	cudaFree((void*)d_index);
	cudaFree((void*)d_index_1);
	cudaFree((void*)d_res_1);
	cudaFree((void*)d_res_2);
	cudaFree((void*)d_index_2);
	cudaDeviceSynchronize();
}