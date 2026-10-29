#pragma once

#ifndef FILTER_H
#define FILTER_H

__global__ void Generate_VI_reference(float* vector_in, int n_X, int n_Y, int n_B, int n_Years, float* img_QA, float* VI_Reference, float* d_distance_similarity, int* d_count_vec_res1/*, double N*/);
__global__ void Compute_similar_pixel(float* vector_in, float* img_QA, float* VI_Reference, int n_X, int n_Y, int n_B, int n_Years, int  win, float* d_res);
__device__ float genfact(int a, int b);
__device__ float gp(int i, int m, int k, int s);
__device__ float weight(int i, int t, int m, int n, int s);
__global__ void STSG_filter(float* vector_in, float* img_QA, float* VI_Reference, float* d_res, int n_X, int n_Y, int n_B, int n_Years, int win, float sampcorr, int StaOrEnd, float* vector_out, int* d_index, int* d_index_1, float* d_res_1, float* d_res_2, int* d_index_2, int m2, int d2, int t, int s);

#endif // !FILTER_H
