#include "Filter.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include <algorithm>

__global__ void Generate_VI_reference(float* vector_in, int n_X, int n_Y, int n_B, int n_Years, float* img_QA, float* VI_Reference, float* d_distance_similarity, int* d_count_vec_res1/*, double N*/)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_X)
		return;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (j >= n_Y)
		return;

	for (int k = 0; k < n_B; k++)
	{
		int count_img_QA = 0;
		float mean_img_QA = 0;
		for (int y = 0; y < n_Years; y++)
		{
			if (vector_in[i + j * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] <= 0.2 || vector_in[i + j * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] >= 1)
				img_QA[i + j * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] = 2;
			else
			{
				count_img_QA++;
				mean_img_QA += vector_in[i + j * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B];
			}
		}
		if (count_img_QA >= 1)
			VI_Reference[i + j * n_X + k * n_X * n_Y] = mean_img_QA / count_img_QA;
	}

	float* distance_similarity = &d_distance_similarity[(i + j * n_X) * (n_B + 2) * n_Years];
	for (int y = 0; y < n_Years; y++)
	{
		int count_img_QA = 0;
		for (int k = 0; k < n_B; k++)
		{
			if (img_QA[i + j * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] == 0)
			{
				double dis = pow((VI_Reference[i + j * n_X + k * n_X * n_Y] - vector_in[i + j * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B]), 2);
				dis = sqrt(dis);

				double xy_sum = vector_in[i + j * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] * VI_Reference[i + j * n_X + k * n_X * n_Y];
				double x2_sum = vector_in[i + j * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] * vector_in[i + j * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B];
				double y2_sum = VI_Reference[i + j * n_X + k * n_X * n_Y] * VI_Reference[i + j * n_X + k * n_X * n_Y];
				if (x2_sum != 0 && y2_sum != 0)
				{
					double sim = xy_sum / sqrt(x2_sum * y2_sum);

					count_img_QA++;
					distance_similarity[k + y * (n_B + 2)] = dis * (1 - sim);
					distance_similarity[n_B + y * (n_B + 2)] += xy_sum / sqrt(x2_sum * y2_sum);
				}
				count_img_QA++;
			}
		}
		distance_similarity[n_B + y * (n_B + 2)] = distance_similarity[n_B + y * (n_B + 2)] / count_img_QA;
		for (int k = 0; k < n_B; k++)
		{
			if (img_QA[i + j * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] == 0 && distance_similarity[k + y * (n_B + 2)] != 0)
				distance_similarity[n_B + 1 + y * (n_B + 2)] += pow((distance_similarity[k + y * (n_B + 2)] - distance_similarity[n_B + y * (n_B + 2)]), 2);
		}
		distance_similarity[n_B + 1 + y * (n_B + 2)] = distance_similarity[n_B + 1 + y * (n_B + 2)] / count_img_QA;
		distance_similarity[n_B + 1 + y * (n_B + 2)] = sqrt(distance_similarity[n_B + 1 + y * (n_B + 2)]);
		distance_similarity[n_B + y * (n_B + 2)] += 0.5*distance_similarity[n_B + 1 + y * (n_B + 2)];
	}

	int count_vec_res1 = 0;
	int* res_vec_res1 = &d_count_vec_res1[(i + j * n_X) * n_B];
	for (int k = 0; k < n_B; k++)
	{
		int count_img_QA = 0;
		float mean_img_QA = 0;
		for (int y = 0; y < n_Years; y++)
		{
			if (img_QA[i + j * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] == 0 && distance_similarity[k + y * (n_B + 2)] < distance_similarity[n_B + y * (n_B + 2)])
			{
				count_img_QA++;
				mean_img_QA += vector_in[i + j * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B];
				img_QA[i + j * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] = 0;
			}
			else
				img_QA[i + j * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] = 2;
		}
		if (count_img_QA >= 1)
		{
			VI_Reference[i + j * n_X + k * n_X * n_Y] = mean_img_QA / count_img_QA;
			res_vec_res1[count_vec_res1++] = k;
		}
		else
			VI_Reference[i + j * n_X + k * n_X * n_Y] = 0;
	}

	if (count_vec_res1 < n_B && count_vec_res1 > 1)
	{
		if (res_vec_res1[count_vec_res1 - 1] != n_B - 1)
		{
			double k = (VI_Reference[i + j * n_X + res_vec_res1[count_vec_res1 - 1] * n_X * n_Y] - VI_Reference[i + j * n_X + res_vec_res1[0] * n_X * n_Y]) / (n_B + res_vec_res1[0] - res_vec_res1[count_vec_res1 - 1]);
			VI_Reference[i + j * n_X + (n_B - 1) * n_X * n_Y] = VI_Reference[i + j * n_X + res_vec_res1[count_vec_res1 - 1] * n_X * n_Y] - k * (n_B - 1 - res_vec_res1[count_vec_res1 - 1]);
			res_vec_res1[count_vec_res1++] = n_B - 1;
		}

		double k = 0;
		int x = 0;
		int l = 0;
		int count_res_vec_res1 = count_vec_res1;
		while (count_vec_res1 < n_B && x < count_res_vec_res1 - 1)
		{
			l = res_vec_res1[x + 1] - res_vec_res1[x];
			int n = 1;
			while (l > 1)
			{
				k = (VI_Reference[i + j * n_X + res_vec_res1[x + 1] * n_X * n_Y] - VI_Reference[i + j * n_X + res_vec_res1[x] * n_X * n_Y]) / (res_vec_res1[x + 1] - res_vec_res1[x]);
				VI_Reference[i + j * n_X + (res_vec_res1[x] + n) * n_X * n_Y] = VI_Reference[i + j * n_X + res_vec_res1[x] * n_X * n_Y] + k * n;
				count_vec_res1++;
				n++;
				l--;
			}
			x++;
		}

		if (res_vec_res1[0] != 0)
		{
			double k = (VI_Reference[i + j * n_X + (n_B - 1) * n_X * n_Y] - VI_Reference[i + j * n_X + res_vec_res1[0] * n_X * n_Y]) / (res_vec_res1[0] + 1);
			int l = res_vec_res1[0];
			int n = 0;
			while (l >= 1)
			{
				VI_Reference[i + j * n_X + n * n_X * n_Y] = VI_Reference[i + j * n_X + res_vec_res1[0] * n_X * n_Y] + k * l;
				n++;
				l--;
			};
		}
	}
}

__global__ void Compute_similar_pixel(float* vector_in, float* img_QA, float* VI_Reference, int n_X, int n_Y, int n_B, int n_Years, int  win, float* d_res)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_X)
		return;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (j >= n_Y)
		return;

	for (int y = 0; y < n_Years; y++)
	{
		int no_VI = 0;
		for (int k = 0; k < n_B; k++)
		{
			if (img_QA[i + j * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] == 0)
			{
				if (no_VI == 0)
				{
					float VI_ref = vector_in[i + j * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] - VI_Reference[i + j * n_X + k * n_X * n_Y];
					for (int nk = 0; nk < k; nk++)
						vector_in[i + j * n_X + nk * n_X * n_Y + y * n_X * n_Y * n_B] = VI_ref + VI_Reference[i + j * n_X + nk * n_X * n_Y];
				}
				else if (img_QA[i + j * n_X + (k - 1) * n_X * n_Y + y * n_X * n_Y * n_B] == 2)
				{
					float VI_ref_1 = vector_in[i + j * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] - VI_Reference[i + j * n_X + k * n_X * n_Y];
					float VI_ref_2 = vector_in[i + j * n_X + (no_VI - 1) * n_X * n_Y + y * n_X * n_Y * n_B] - VI_Reference[i + j * n_X + (no_VI - 1) * n_X * n_Y];
					float VI_ref = (VI_ref_1 - VI_ref_2) / (k - no_VI + 1);
					for (int nk = no_VI; nk < k; nk++)
						vector_in[i + j * n_X + nk * n_X * n_Y + y * n_X * n_Y * n_B] = VI_ref_2 + VI_ref * (nk - no_VI + 1) + VI_Reference[i + j * n_X + nk * n_X * n_Y];
				}
				no_VI = k + 1;
			}
		}
		if (no_VI != n_B && no_VI != 0)
		{
			float VI_ref = vector_in[i + j * n_X + (no_VI - 1) * n_X * n_Y + y * n_X * n_Y * n_B] - VI_Reference[i + j * n_X + (no_VI - 1) * n_X * n_Y];
			for (; no_VI < n_B; no_VI++)
				vector_in[i + j * n_X + no_VI * n_X * n_Y + y * n_X * n_Y * n_B] = VI_ref + VI_Reference[i + j * n_X + no_VI * n_X * n_Y];
		}
	}

	float* Slope_res = &d_res[(i + j * n_X) * (2 * win + 1) * (2 * win + 1) * 3];
	float* Intercept_res = &d_res[(i + j * n_X) * (2 * win + 1) * (2 * win + 1) * 3 + (2 * win + 1) * (2 * win + 1)];
	float* new_corr_similar_res = &d_res[(i + j * n_X) * (2 * win + 1) * (2 * win + 1) * 3 + (2 * win + 1) * (2 * win + 1) * 2];
	for (int sj = -1 * win; sj <= win; sj++)
	{
		for (int si = -1 * win; si <= win; si++)
		{
			if (i + si >= 0 && i + si < n_X && j + sj >= 0 && j + sj < n_Y)
			{
				int count_tempQA = 0;
				double x_sum = 0;
				double y_sum = 0;
				double x_mean = 0;
				double y_mean = 0;
				double xy_sum = 0;
				double x2_sum = 0;
				double y2_sum = 0;
				for (int k = 0; k < n_B; k++)
				{
					for (int y = 0; y < n_Years; y++)
					{
						if (VI_Reference[i + (j)*n_X + k * n_X * n_Y] != 0 && VI_Reference[(i + si) + (j + sj) * n_X + k * n_X * n_Y] != 0)
						{
							count_tempQA++;
							x_sum += vector_in[i + si + (j + sj) * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] / VI_Reference[(i + si) + (j + sj) * n_X + k * n_X * n_Y];
							y_sum += vector_in[i + (j)*n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] / VI_Reference[i + (j)*n_X + k * n_X * n_Y];
							xy_sum += (vector_in[i + si + (j + sj) * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] / VI_Reference[(i + si) + (j + sj) * n_X + k * n_X * n_Y] - x_mean) * (vector_in[i + (j)*n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] / VI_Reference[i + (j)*n_X + k * n_X * n_Y] - y_mean);
							x2_sum += (vector_in[i + si + (j + sj) * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] / VI_Reference[(i + si) + (j + sj) * n_X + k * n_X * n_Y] - x_mean) * (vector_in[i + si + (j + sj) * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] / VI_Reference[(i + si) + (j + sj) * n_X + k * n_X * n_Y] - x_mean);
							y2_sum += (vector_in[i + (j)*n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] / VI_Reference[i + (j)*n_X + k * n_X * n_Y] - y_mean) * (vector_in[i + (j)*n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] / VI_Reference[i + (j)*n_X + k * n_X * n_Y] - y_mean);
						}
					}
				}
				if (count_tempQA >= 30)
				{
					Slope_res[si + win + (sj + win) * (2 * win + 1)] = (xy_sum * count_tempQA - x_sum * y_sum) / (x2_sum * count_tempQA - x_sum * x_sum);
					Intercept_res[si + win + (sj + win) * (2 * win + 1)] = (x2_sum * y_sum - x_sum * xy_sum) / (x2_sum * count_tempQA - x_sum * x_sum);

					x_mean = x_sum / count_tempQA;
					y_mean = y_sum / count_tempQA;
					xy_sum = 0;
					x2_sum = 0;
					y2_sum = 0;
					for (int k = 0; k < n_B; k++)
					{
						for (int y = 0; y < n_Years; y++)
						{
							if (VI_Reference[i + (j)*n_X + k * n_X * n_Y] != 0 && VI_Reference[(i + si) + (j + sj) * n_X + k * n_X * n_Y] != 0)
							{
								xy_sum += (vector_in[i + si + (j + sj) * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] / VI_Reference[(i + si) + (j + sj) * n_X + k * n_X * n_Y] - x_mean) * (vector_in[i + (j)*n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] / VI_Reference[i + (j)*n_X + k * n_X * n_Y] - y_mean);
								x2_sum += (vector_in[i + si + (j + sj) * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] / VI_Reference[(i + si) + (j + sj) * n_X + k * n_X * n_Y] - x_mean) * (vector_in[i + si + (j + sj) * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] / VI_Reference[(i + si) + (j + sj) * n_X + k * n_X * n_Y] - x_mean);
								y2_sum += (vector_in[i + (j)*n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] / VI_Reference[i + (j)*n_X + k * n_X * n_Y] - y_mean) * (vector_in[i + (j)*n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] / VI_Reference[i + (j)*n_X + k * n_X * n_Y] - y_mean);
							}
						}
					}
					new_corr_similar_res[si + win + (sj + win) * (2 * win + 1)] = xy_sum / sqrt(x2_sum * y2_sum);
				}
			}
		}
	}
}

// generalized factorial (a)(a-1)...(a-b+1)
__device__ float genfact(int a, int b)
{
	float gf = 1.0;
	for (int j = (a - b) + 1; j <= a; j++)
		gf *= j;
	return gf;
}

__device__ float gp(int i, int m, int k, int s)
{
	if (k > 0)
		return (4.0 * k - 2.0) / (k * (2.0 * m - k + 1.0)) * (i * gp(i, m, k - 1, s) +
			s * gp(i, m, k - 1, s - 1)) - ((k - 1.0) * (2.0 * m + k)) / (k * (2.0 * m - k + 1.0)) * gp(i, m, k - 2, s);
	else
	{
		if (k == 0 && s == 0)
			return 1.0;
		else
			return 0.0;
	}
}

// calculate the weight of the ith dat point for the tth lease-square point of the sth derivative, over 2m+1 points and order n
__device__ float weight(int i, int t, int m, int n, int s)
{
	float w = 0;
	for (int k = 0; k <= n; k++)
		w += (2 * k + 1) * (genfact(2 * m, k) / genfact(2 * m + k + 1, k + 1)) * gp(i, m, k, 0) * gp(t, m, k, s);
	return w;
}

__global__ void STSG_filter(float* vector_in, float* img_QA, float* VI_Reference, float* d_res, int n_X, int n_Y, int n_B, int n_Years, int win, float sampcorr, int StaOrEnd, float* vector_out, int* d_index, int* d_index_1, float* d_res_1, float* d_res_2, int* d_index_2, int m2, int d2, int t, int s)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_X)
		return;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (StaOrEnd == 0 && (j >= n_Y - win || j < win))
		return;
	if (StaOrEnd == 1 && j >= n_Y - win)
		return;
	if (StaOrEnd == 2 && (j >= n_Y || j < win))
		return;

	for (int y = 0; y < n_Years; y++)
	{
		float* corr_res = &d_res_1[(i + j * n_X) * (2 * win + 1) * (2 * win + 1) * 4];
		for (int sj = -1 * win; sj <= win; sj++)
		{
			for (int si = -1 * win; si <= win; si++)
			{
				if (i + si >= 0 && i + si < n_X && j + sj >= 0 && j + sj < n_Y)
				{
					double x_sum = 0;
					double y_sum = 0;
					double x_mean = 0;
					double y_mean = 0;
					double xy_sum = 0;
					double x2_sum = 0;
					double y2_sum = 0;
					for (int k = 0; k < n_B; k++)
					{
						x_sum += vector_in[(i + si) + (j + sj) * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B];
						y_sum += vector_in[i + (j)*n_X + k * n_X * n_Y + y * n_X * n_Y * n_B];
					}
					x_mean = x_sum / n_B;
					y_mean = y_sum / n_B;
					for (int k = 0; k < n_B; k++)
					{
						xy_sum += (vector_in[i + si + (j + sj) * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] - x_mean) * (vector_in[i + (j)*n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] - y_mean);
						x2_sum += (vector_in[i + si + (j + sj) * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] - x_mean) * (vector_in[(i + si) + (j + sj) * n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] - x_mean);
						y2_sum += (vector_in[i + (j)*n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] - y_mean) * (vector_in[i + (j)*n_X + k * n_X * n_Y + y * n_X * n_Y * n_B] - y_mean);
					}
					corr_res[si + win + (sj + win) * (2 * win + 1)] = xy_sum / sqrt(x2_sum * y2_sum);
				}
			}
		}

		float* Slope_res = &d_res[(i + j * n_X) * (2 * win + 1) * (2 * win + 1) * 3];
		float* Intercept_res = &d_res[(i + j * n_X) * (2 * win + 1) * (2 * win + 1) * 3 + (2 * win + 1) * (2 * win + 1)];
		float* new_corr_similar_res = &d_res[(i + j * n_X) * (2 * win + 1) * (2 * win + 1) * 3 + (2 * win + 1) * (2 * win + 1) * 2];
		int count_corr_Slope = 0;
		int samp =win * win;
		int* similar_index = &d_index_1[(i + j * n_X) * (2 * win + 1) * (2 * win + 1) * 2];
		float* slope_intercept = &d_res_1[(i + j * n_X) * (2 * win + 1) * (2 * win + 1) * 4 + (2 * win + 1) * (2 * win + 1)];
		float* corr_similar = &d_res_1[(i + j * n_X) * (2 * win + 1) * (2 * win + 1) * 4 + (2 * win + 1) * (2 * win + 1) * 3];
		for (int m = 0; m < (2 * win + 1) * (2 * win + 1); m++)
		{
			if (corr_res[m] >= sampcorr && Slope_res[m] != 0.)
				count_corr_Slope++;
		}
		if (count_corr_Slope >= samp)
		{
			int* new_corr = &d_index[(i + j * n_X) * (2 * win + 1) * (2 * win + 1)];
			for (int m = 0; m < (2 * win + 1) * (2 * win + 1); m++)
				new_corr[m] = m;
			for (int m = 0; m < count_corr_Slope; m++)
			{
				for (int n = m + 1; n < (2 * win + 1) * (2 * win + 1); n++)
				{
					if (corr_res[new_corr[m]] < corr_res[new_corr[n]])
					{
						int temp = new_corr[m];
						new_corr[m] = new_corr[n];
						new_corr[n] = temp;
					}
				}
			}

			for (int k = 0; k < samp; k++)
			{
				similar_index[1 + k * 2] = int(new_corr[k + 1] / (2 * win + 1)) + j - win;
				similar_index[k * 2] = new_corr[k + 1] - int(new_corr[k + 1] / (2 * win + 1)) * (2 * win + 1) + i - win;
				slope_intercept[1 + k * 2] = Slope_res[int(new_corr[k + 1] / (2 * win + 1)) + (2 * win + 1) * (new_corr[k + 1] - int(new_corr[k + 1] / (2 * win + 1)) * (2 * win + 1))];
				slope_intercept[k * 2] = Intercept_res[int(new_corr[k + 1] / (2 * win + 1)) + (2 * win + 1) * (new_corr[k + 1] - int(new_corr[k + 1] / (2 * win + 1)) * (2 * win + 1))];
				corr_similar[k] = new_corr_similar_res[int(new_corr[k + 1] / (2 * win + 1)) + (2 * win + 1) * (new_corr[k + 1] - int(new_corr[k + 1] / (2 * win + 1)) * (2 * win + 1))];
			}
		}
		else
		{
			samp =count_corr_Slope;
			int* new_corr = &d_index[(i + j * n_X) * (2 * win + 1) * (2 * win + 1)];
			for (int m = 0; m < (2 * win + 1) * (2 * win + 1); m++)
				new_corr[m] = m;
			for (int m = 0; m < count_corr_Slope; m++)
			{
				for (int n = m + 1; n < (2 * win + 1) * (2 * win + 1); n++)
				{
					if (corr_res[new_corr[m]] < corr_res[new_corr[n]])
					{
						int temp = new_corr[m];
						new_corr[m] = new_corr[n];
						new_corr[n] = temp;
					}
				}
			}

			for (int k = 0; k < samp; k++)
			{
				similar_index[1 + k * 2] = int(new_corr[k + 1] / (2 * win + 1)) + j - win;
				similar_index[k * 2] = new_corr[k + 1] - int(new_corr[k + 1] / (2 * win + 1)) * (2 * win + 1) + i - win;
				slope_intercept[1 + k * 2] = Slope_res[int(new_corr[k + 1] / (2 * win + 1)) + (2 * win + 1) * (new_corr[k + 1] - int(new_corr[k + 1] / (2 * win + 1)) * (2 * win + 1))];
				slope_intercept[k * 2] = Intercept_res[int(new_corr[k + 1] / (2 * win + 1)) + (2 * win + 1) * (new_corr[k + 1] - int(new_corr[k + 1] / (2 * win + 1)) * (2 * win + 1))];
				corr_similar[k] = new_corr_similar_res[int(new_corr[k + 1] / (2 * win + 1)) + (2 * win + 1) * (new_corr[k + 1] - int(new_corr[k + 1] / (2 * win + 1)) * (2 * win + 1))];
			}
		}

		float* trend_VI = &d_res_2[(i + j * n_X) * n_B*4];
		int count_trend_VI = 0;
		int* res_trend_VI = &d_index_2[(i + j * n_X) * n_B];
		for (int k = 0; k < n_B; k++)
		{
			float* temp_VI = &d_res_1[(i + j * n_X) * (2 * win + 1) * (2 * win + 1) * 4];
			int count_temp_VI = 0;
			float total_new_corr_similar = 0;
			float total_new_temp = 0;
			for (int m = 0; m < samp; m++)
			{
				if (similar_index[m * 2] >= 0 && similar_index[m * 2] < n_X && similar_index[m * 2 + 1] >= 0 && similar_index[m * 2 + 1] < n_Y
					&& VI_Reference[similar_index[m * 2] + (similar_index[1 + m * 2]) * n_X + k * n_X * (n_Y)] != 0)
				{
					float new_ratio = vector_in[similar_index[m * 2] + (similar_index[1 + m * 2]) * n_X + k * n_X * (n_Y)+y * n_X * (n_Y)*n_B] / VI_Reference[similar_index[m * 2] + (similar_index[1 + m * 2]) * n_X + k * n_X * (n_Y)];
					temp_VI[m] = (slope_intercept[m * 2] + new_ratio * slope_intercept[1 + m * 2]) * VI_Reference[i + (j)*n_X + k * n_X * (n_Y)];
					if (temp_VI[m] >= 1 || temp_VI[m] <= -0.2)
						temp_VI[m] = 0;
					if (temp_VI[m] != 0)
					{
						count_temp_VI++;
						total_new_corr_similar += corr_similar[m];
					}
				}
				else
					temp_VI[m] = 0;
			}
			if (count_temp_VI != 0)
			{
				for (int m = 0; m < samp; m++)
				{
					if (temp_VI[m] != 0)
						total_new_temp += corr_similar[m] / total_new_corr_similar * temp_VI[m];
				}
				trend_VI[k] = total_new_temp;
				if (trend_VI[k] != 0)
					res_trend_VI[count_trend_VI++] = k;
			}
			else
				trend_VI[k] = 0;
		}

		if (count_trend_VI < n_B && count_trend_VI > 1)
		{
			if (res_trend_VI[count_trend_VI - 1] != n_B - 1)
			{
				double k = (trend_VI[res_trend_VI[count_trend_VI - 1]] - trend_VI[res_trend_VI[0]]) / (n_B + res_trend_VI[0] - res_trend_VI[count_trend_VI - 1]);
				trend_VI[n_B - 1] = trend_VI[res_trend_VI[count_trend_VI - 1]] - k * (n_B - 1 - res_trend_VI[count_trend_VI - 1]);
				res_trend_VI[count_trend_VI++] = n_B - 1;
			}

			double k = 0;
			int x = 0;
			int l = 0;
			int count_res_vec_res1 = count_trend_VI;
			while (count_trend_VI < n_B && x < count_res_vec_res1 - 1)
			{
				l = res_trend_VI[x + 1] - res_trend_VI[x];
				int n = 1;
				while (l > 1)
				{
					k = (trend_VI[res_trend_VI[x + 1]] - trend_VI[res_trend_VI[x]]) / (res_trend_VI[x + 1] - res_trend_VI[x]);
					trend_VI[res_trend_VI[x] + n] = trend_VI[res_trend_VI[x]] + k * n;
					count_trend_VI++;
					n++;
					l--;
				}
				x++;
			}

			if (res_trend_VI[0] != 0)
			{
				double k = (trend_VI[n_B - 1] - trend_VI[res_trend_VI[0]]) / (res_trend_VI[0] + 1);
				int l = res_trend_VI[0];
				int n = 0;
				while (l >= 1)
				{
					trend_VI[n] = trend_VI[res_trend_VI[0]] + k * l;
					n++;
					l--;
				};
			}
		}

		float* fl = &d_res_2[(i + j * n_X) * n_B * 4 + n_B];
		int count_fl = 0;
		float mean_fl = 0;
		for (int k = 0; k < n_B; k++)
		{
			if (img_QA[i + (j)*n_X + k * n_X * (n_Y)+y * n_X * (n_Y)*n_B] == 0)
			{
				fl[k] = vector_in[i + (j)*n_X + k * n_X * (n_Y)+y * n_X * (n_Y)*n_B] - trend_VI[k];
				count_fl++;
				mean_fl += fl[k];
			}
			else
				fl[k] = -1.0;
		}
		if (count_fl != 0)
			mean_fl = mean_fl / count_fl;

		float gdis = 0;
		float min_fl = 0;
		float max_fl = 0;
		for (int k = 0; k < n_B; k++)
		{
			if (min_fl > fl[k])
				min_fl = fl[k];
			if (max_fl < fl[k])
				max_fl = fl[k];
		}
		for (int k = 0; k < n_B; k++)
		{
			if (fl[k] == -1.0)
				fl[k] = min_fl;
		}
		for (int k = 0; k < n_B; k++)
		{
			fl[k] = (fl[k] - min_fl) / (max_fl - min_fl);
			gdis += fl[k] * abs(vector_in[i + (j)*n_X + k * n_X * (n_Y)+y * n_X * (n_Y)*n_B] - trend_VI[k]);
		}

		for (int k = 0; k < n_B; k++)
		{
			if (vector_in[i + (j)*n_X + k * n_X * (n_Y)+y * n_X * (n_Y)*n_B] > trend_VI[k])
				trend_VI[k] = vector_in[i + (j)*n_X + k * n_X * (n_Y)+y * n_X * (n_Y)*n_B];
			else
				vector_in[i + (j)*n_X + k * n_X * (n_Y)+y * n_X * (n_Y)*n_B] = trend_VI[k];
		}

		float ormax = gdis;
		int loop_times = 0;
		float* savgolFilter_fine = &d_res_2[(i + j * n_X) * n_B * 4 + n_B*2];	//set the window width(4, 4) and degree(6) for repetition
		for (int i = 0; i < 2 * m2 + 1; i++)
			savgolFilter_fine[i] = weight(i - m2, t, m2, d2, s);
		float* img_VI_filter = &d_res_2[(i + j * n_X) * n_B * 4 + n_B * 3];

		while (loop_times < 1000)
		{
			loop_times = loop_times + 1;
			ormax = gdis;
			gdis = 0.0;

			for (int k = 0; k < n_B; k++)
			{
				img_VI_filter[k] = trend_VI[k] * savgolFilter_fine[m2];

				for (int n = 1; n <= m2; n++)
				{
					if (k < n)
						img_VI_filter[k] += trend_VI[0] * savgolFilter_fine[m2 - n];
					else
						img_VI_filter[k] += trend_VI[(k - n)] * savgolFilter_fine[m2 - n];

					if (k + n >= n_B)
						img_VI_filter[k] += trend_VI[(n_B - 1)] * savgolFilter_fine[m2 + n];
					else
						img_VI_filter[k] += trend_VI[(k + n)] * savgolFilter_fine[m2 + n];
				}
				gdis += fl[k] * abs(img_VI_filter[k] - vector_in[i + (j)*n_X + k * n_X * (n_Y)+y * n_X * (n_Y)*n_B]);
			}

			if (StaOrEnd == 1)
			{
				for (int k = 0; k < n_B; k++)
					vector_out[i + j * n_X + k * n_X * (n_Y - win) + y * n_X * (n_Y - win) * n_B] = trend_VI[k];
			}
			else if (StaOrEnd == 2)
			{
				for (int k = 0; k < n_B; k++)
					vector_out[i + (j - win) * n_X + k * n_X * (n_Y - win) + y * n_X * (n_Y - win) * n_B] = trend_VI[k];
			}
			else
			{
				for (int k = 0; k < n_B; k++)
					vector_out[i + (j - win) * n_X + k * n_X * (n_Y - 2 * win) + y * n_X * (n_Y - 2 * win) * n_B] = trend_VI[k];
			}

			if (gdis > ormax)
				break;

			for (int k = 0; k < n_B; k++)
			{
				if (img_VI_filter[k] > vector_in[i + (j)*n_X + k * n_X * (n_Y)+y * n_X * (n_Y)*n_B])
					trend_VI[k] = img_VI_filter[k];
			}
		}

		if (StaOrEnd == 1)
		{
			for (int k = 0; k < n_B; k++)
				trend_VI[k] = vector_out[i + j * n_X + k * n_X * (n_Y - win) + y * n_X * (n_Y - win) * n_B];
			for (int smi = 0; smi < n_B - 4; smi++)
			{
				float a1 = trend_VI[smi];
				float a2 = trend_VI[smi + 1];
				float a3 = trend_VI[smi + 2];
				float a4 = trend_VI[smi + 3];
				float a5 = trend_VI[smi + 4];
				if ((a1 > a2) && (a2 < a3) && (a3 > a4) && (a4 < a5))
				{
					vector_out[i + j * n_X + (smi + 1) * n_X * (n_Y - win) + y * n_X * (n_Y - win) * n_B] = (a1 + a3) / 2.0;
					vector_out[i + j * n_X + (smi + 3) * n_X * (n_Y - win) + y * n_X * (n_Y - win) * n_B] = (a3 + a5) / 2.0;
				}
			}
		}
		else if (StaOrEnd == 2)
		{
			for (int k = 0; k < n_B; k++)
				trend_VI[k] = vector_out[i + (j - win) * n_X + k * n_X * (n_Y - win) + y * n_X * (n_Y - win) * n_B];
			for (int smi = 0; smi < n_B - 4; smi++)
			{
				float a1 = trend_VI[smi];
				float a2 = trend_VI[smi + 1];
				float a3 = trend_VI[smi + 2];
				float a4 = trend_VI[smi + 3];
				float a5 = trend_VI[smi + 4];
				if ((a1 > a2) && (a2 < a3) && (a3 > a4) && (a4 < a5))
				{
					vector_out[i + (j - win) * n_X + (smi + 1) * n_X * (n_Y - win) + y * n_X * (n_Y - win) * n_B] = (a1 + a3) / 2.0;
					vector_out[i + (j - win) * n_X + (smi + 3) * n_X * (n_Y - win) + y * n_X * (n_Y - win) * n_B] = (a3 + a5) / 2.0;
				}
			}
		}
		else
		{
			for (int k = 0; k < n_B; k++)
				trend_VI[k] = vector_out[i + (j - win) * n_X + k * n_X * (n_Y - 2 * win) + y * n_X * (n_Y - 2 * win) * n_B];
			for (int smi = 0; smi < n_B - 4; smi++)
			{
				float a1 = trend_VI[smi];
				float a2 = trend_VI[smi + 1];
				float a3 = trend_VI[smi + 2];
				float a4 = trend_VI[smi + 3];
				float a5 = trend_VI[smi + 4];
				if ((a1 > a2) && (a2 < a3) && (a3 > a4) && (a4 < a5))
				{
					vector_out[i + (j - win) * n_X + (smi + 1) * n_X * (n_Y - 2 * win) + y * n_X * (n_Y - 2 * win) * n_B] = (a1 + a3) / 2.0;
					vector_out[i + (j - win) * n_X + (smi + 3) * n_X * (n_Y - 2 * win) + y * n_X * (n_Y - 2 * win) * n_B] = (a3 + a5) / 2.0;
				}
			}
		}
	}
}

