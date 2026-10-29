#include "newSTSG.h"

#include "gdal_priv.h"

#include <mpi.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include<iostream>
using namespace std;

//******************************************************************
//Input parameters
//the years of the data
const int Years[] = { /*2014,2015,2016,2017,2018,*/2019, 2020, 2021, 2022/**/ };

//the path of the VI data
string VI_path = "../EVI2/EVI2_";//low_zone_3_

//the half width of neighborhood
const int win = 5;

//the thereshold of correlation coefficient to define similar pixels
const float sampcorr = 0.95;

//the parameters for the iterative s - g filtering
//const int m2 = 4;	//set the window width 2 * m2 + 1 for the iterative s - g filtering
//const int d2 = 6;	//set the degree d2 for the iterative s - g filtering
const int tt = 0;	//how to center the estimate 
const int ss = 0;	//0=smoothing, 1=first_derivative ...

//the path of the reconstructed VI data
//string re_VI_path = "../EVI2_newSTSG.tif";//low.tif

int main(int argc, char* argv[])
{
	int myid, numprocs;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	GDALAllRegister();
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	GDALDriver* pDriver = GetGDALDriverManager()->GetDriverByName("GTIFF");
	char** ppszOptions = NULL;
	ppszOptions = CSLSetNameValue(ppszOptions, "BIGTIFF", "IF_NEEDED");

		////parameters
		//if (argc != 2)
		//{
		//	cout << "No parameter file!" << endl;
		//	return 1;
		//}
	
		//ifstream parameter(argv[1]);
		//if (!parameter)
		//{
		//	cout << "Can't open parameter file!" << endl;
		//	return 1;
		//}
	
		//int* Years = nullptr;
		//string NDVI_path, Reliability_path, STSG_Test_path;
		//float cosyear, sampcorr;
		//int win_year, win, snow_address, n_Years;
		//string par;
		//while (getline(parameter, par))
		//{
		//	if (par.substr(0, 2) == "//" || par == "")
		//		continue;
	
		//	for (int i = 0; i < par.size(); )
		//	{
		//		if (isspace(par[i]))
		//			par.erase(i,1);
		//		else
		//			i++;
		//	}
		//	if (par.substr(0, par.find("=")) == "Years")
		//	{
		//		vector<int> year;
		//		while (par.rfind(",") < par.size())
		//		{
		//			year.push_back(stoi(par.substr(par.rfind(",") + 1)));
		//			par = par.substr(0, par.rfind(","));
		//		}
		//		year.push_back(stoi(par.substr(par.rfind("=") + 1)));
	
		//		n_Years = year.size();
		//		Years = new int[n_Years];
		//		for (int i = 0; i < n_Years; i++)
		//			Years[i] = year[n_Years - i - 1];
		//	}
		//	else if (par.substr(0, par.find("=")) == "NDVI_path")
		//		NDVI_path = par.substr(par.find("=") + 1);
		//	else if (par.substr(0, par.find("=")) == "Reliability_path")
		//		Reliability_path = par.substr(par.find("=") + 1);
		//	else if (par.substr(0, par.find("=")) == "STSG_Test_path")
		//		STSG_Test_path = par.substr(par.find("=") + 1);
		//	else if (par.substr(0, par.find("=")) == "cosyear")
		//		cosyear = stof(par.substr(par.find("=") + 1));
		//	else if (par.substr(0, par.find("=")) == "win_year")
		//		win_year = stoi(par.substr(par.find("=") + 1));
		//	else if (par.substr(0, par.find("=")) == "win")
		//		win = stoi(par.substr(par.find("=") + 1));
		//	else if (par.substr(0, par.find("=")) == "sampcorr")
		//		sampcorr = stof(par.substr(par.find("=") + 1));
		//	else if (par.substr(0, par.find("=")) == "snow_address")
		//		snow_address = stoi(par.substr(par.find("=") + 1));
		//}
		//parameter.close();

	////double N[7] = { 0.5,1,1.5,2,2.5,3,5 };
	////for (int nn = 0; nn < 7; nn++)
	////{
	//for (int mm = 6; mm < 7; mm++)
	//{
	//	for (int dd = 6; dd < 8; dd++)
	//	{
	////int mm = 0, dd = 0;
	//		
	//int* Time = new int[5] {0};//(Others;Reading;Transfering;Computing;Writing;)
	//		time_t t[2] = { 0 };
	//		time(&t[0]);

	//		time_t T[2] = { 0 };
	//		time(&T[0]);

	//		int n_Years = 0, n_X = 0, n_Y = 0, n_B = 0, n_Device = 0, ReadLoop = 0, PerStep = 0;
	//		float* vector_in;
	//		GDALDataType type_VI;
	//		if (myid == 0)
	//		{
	//			n_Years = sizeof(Years) / sizeof(Years[0]);

	//			string FileName = VI_path + to_string(Years[0]);
	//			GDALDataset* File = (GDALDataset*)GDALOpen(FileName.c_str(), GA_ReadOnly);
	//			n_X = File->GetRasterXSize();
	//			n_Y = File->GetRasterYSize();
	//			n_B = File->GetRasterCount();
	//			type_VI = File->GetRasterBand(1)->GetRasterDataType();
	//			GDALClose(File);

	//			cudaGetDeviceCount(&n_Device);
	//			cudaDeviceProp prop;
	//			for (int i = 0; i < n_Device; i++)
	//			{
	//				cudaGetDeviceProperties(&prop, i);
	//				cout << "Device " << i << "'s Property:" << endl;
	//				cout << "Name:" << prop.name << endl;
	//				cout << "TotalGlobalMem:" << prop.totalGlobalMem / 1024 / 1024 << " MB" << endl;
	//				cout << "SharedMemPerBlock:" << prop.sharedMemPerBlock / 1024 << " KB" << endl;
	//				cout << "MaxThreadsPerBlock:" << prop.maxThreadsPerBlock << endl;
	//				cout << "MaxGridSize:" << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << endl;
	//				cout << "WarpSize:" << prop.warpSize << endl;
	//			}

	//			size_t PerYSize = (1.2) * (n_X * (n_B + 2) * n_Years * (4 * sizeof(float)) + n_X * n_B * (2 * sizeof(int) + 5 * sizeof(float)) + n_X * (2 * win + 1) * (2 * win + 1) * (7 * sizeof(float) + 3 * sizeof(int)));
	//			if (prop.totalGlobalMem <= PerYSize * (2 * win + 1))
	//			{
	//				cout << "Serial is better!" << endl;
	//				return 1;
	//			}
	//			if (n_Device > 1)
	//			{
	//				size_t perStep = prop.totalGlobalMem / PerYSize;
	//				ReadLoop = 1;
	//				if (perStep < n_Y)
	//				{
	//					ReadLoop = n_Y / perStep + 1;
	//					perStep = n_Y / ReadLoop + 1;
	//				}

	//				if (ReadLoop < n_Device)
	//				{
	//					ReadLoop = n_Device;
	//					perStep = n_Y / ReadLoop + 1;
	//				}
	//				PerStep = perStep + 2 * win;
	//			}
	//			else
	//			{
	//				size_t perStep = prop.totalGlobalMem / PerYSize;
	//				ReadLoop = 1;
	//				if (perStep < n_Y)
	//				{
	//					ReadLoop = n_Y / perStep + 1;
	//					perStep = n_Y / ReadLoop + 1;
	//				}
	//				PerStep = perStep + 2 * win;
	//			}
	//		}
	//		MPI_Bcast(&n_Years, 1, MPI_INT, 0, MPI_COMM_WORLD);
	//		MPI_Bcast(&n_X, 1, MPI_INT, 0, MPI_COMM_WORLD);
	//		MPI_Bcast(&n_Y, 1, MPI_INT, 0, MPI_COMM_WORLD);
	//		MPI_Bcast(&n_B, 1, MPI_INT, 0, MPI_COMM_WORLD);
	//		MPI_Bcast(&n_Device, 1, MPI_INT, 0, MPI_COMM_WORLD);
	//		MPI_Bcast(&ReadLoop, 1, MPI_INT, 0, MPI_COMM_WORLD);
	//		MPI_Bcast(&PerStep, 1, MPI_INT, 0, MPI_COMM_WORLD);
	//		time(&t[1]);
	//		Time[0] += t[1] - t[0];
	//		t[0] = t[1];

	//		if (myid == 0)
	//		{
	//			for (int rl = 0, StartY = 0; rl < ReadLoop && StartY < n_Y; StartY = (rl + 1) * (PerStep - 2 * win), rl++)
	//			{
	//				vector_in = new float[n_X * PerStep * n_B * n_Years];

	//				if (rl == ReadLoop - 1)
	//					PerStep = n_Y - StartY;

	//				switch (type_VI)
	//				{
	//				case GDT_Byte:
	//				{
	//					unsigned char* vector_inn = new unsigned char[n_X * PerStep * n_B * n_Years];
	//					for (int y = 0; y < n_Years; y++)
	//					{
	//						string FileName = VI_path + to_string(Years[y]);
	//						GDALDataset* File = (GDALDataset*)GDALOpen(FileName.c_str(), GA_ReadOnly);
	//						File->RasterIO(GF_Read, 0, StartY, n_X, PerStep, &vector_inn[n_X * PerStep * n_B * y], n_X, PerStep, type_VI, n_B, nullptr, 0, 0, 0);
	//						GDALClose(File);
	//					}
	//					for (int i = 0; i < n_X * PerStep * n_B * n_Years; i++)
	//						vector_in[i] = float(vector_inn[i]) / 10000;
	//					delete[] vector_inn;
	//					break;
	//				}
	//				case GDT_UInt16:
	//				{
	//					unsigned short* vector_inn = new unsigned short[n_X * PerStep * n_B * n_Years];
	//					for (int y = 0; y < n_Years; y++)
	//					{
	//						string FileName = VI_path + to_string(Years[y]);
	//						GDALDataset* File = (GDALDataset*)GDALOpen(FileName.c_str(), GA_ReadOnly);
	//						File->RasterIO(GF_Read, 0, StartY, n_X, PerStep, &vector_inn[n_X * PerStep * n_B * y], n_X, PerStep, type_VI, n_B, nullptr, 0, 0, 0);
	//						GDALClose(File);
	//					}
	//					for (int i = 0; i < n_X * PerStep * n_B * n_Years; i++)
	//						vector_in[i] = float(vector_inn[i]) / 10000;
	//					delete[] vector_inn;
	//					break;
	//				}
	//				case GDT_Int16:
	//				{
	//					short* vector_inn = new short[n_X * PerStep * n_B * n_Years];
	//					for (int y = 0; y < n_Years; y++)
	//					{
	//						string FileName = VI_path + to_string(Years[y]);
	//						GDALDataset* File = (GDALDataset*)GDALOpen(FileName.c_str(), GA_ReadOnly);
	//						File->RasterIO(GF_Read, 0, StartY, n_X, PerStep, &vector_inn[n_X * PerStep * n_B * y], n_X, PerStep, type_VI, n_B, nullptr, 0, 0, 0);
	//						GDALClose(File);
	//					}
	//					for (int i = 0; i < n_X * PerStep * n_B * n_Years; i++)
	//						vector_in[i] = float(vector_inn[i]) / 10000;
	//					delete[] vector_inn;
	//					break;
	//				}
	//				case GDT_UInt32:
	//				{
	//					unsigned long* vector_inn = new unsigned long[n_X * PerStep * n_B * n_Years];
	//					for (int y = 0; y < n_Years; y++)
	//					{
	//						string FileName = VI_path + to_string(Years[y]);
	//						GDALDataset* File = (GDALDataset*)GDALOpen(FileName.c_str(), GA_ReadOnly);
	//						File->RasterIO(GF_Read, 0, StartY, n_X, PerStep, &vector_inn[n_X * PerStep * n_B * y], n_X, PerStep, type_VI, n_B, nullptr, 0, 0, 0);
	//						GDALClose(File);
	//					}
	//					for (int i = 0; i < n_X * PerStep * n_B * n_Years; i++)
	//						vector_in[i] = float(vector_inn[i]) / 10000;
	//					delete[] vector_inn;
	//					break;
	//				}
	//				case GDT_Int32:
	//				{
	//					long* vector_inn = new long[n_X * PerStep * n_B * n_Years];
	//					for (int y = 0; y < n_Years; y++)
	//					{
	//						string FileName = VI_path + to_string(Years[y]);
	//						GDALDataset* File = (GDALDataset*)GDALOpen(FileName.c_str(), GA_ReadOnly);
	//						File->RasterIO(GF_Read, 0, StartY, n_X, PerStep, &vector_inn[n_X * PerStep * n_B * y], n_X, PerStep, type_VI, n_B, nullptr, 0, 0, 0);
	//						GDALClose(File);
	//					}
	//					for (int i = 0; i < n_X * PerStep * n_B * n_Years; i++)
	//						vector_in[i] = float(vector_inn[i]) / 10000;
	//					delete[] vector_inn;
	//					break;
	//				}
	//				case GDT_Float32:
	//				{
	//					for (int y = 0; y < n_Years; y++)
	//					{
	//						string FileName = VI_path + to_string(Years[y]);
	//						GDALDataset* File = (GDALDataset*)GDALOpen(FileName.c_str(), GA_ReadOnly);
	//						File->RasterIO(GF_Read, 0, StartY, n_X, PerStep, &vector_in[n_X * PerStep * n_B * y], n_X, PerStep, type_VI, n_B, nullptr, 0, 0, 0);
	//						GDALClose(File);
	//					}
	//					for (int i = 0; i < n_X * PerStep * n_B * n_Years; i++)
	//						vector_in[i] = float(vector_in[i]) / 10000;
	//					break;
	//				}
	//				case GDT_Float64:
	//				{
	//					double* vector_inn = new double[n_X * PerStep * n_B * n_Years];
	//					for (int y = 0; y < n_Years; y++)
	//					{
	//						string FileName = VI_path + to_string(Years[y]);
	//						GDALDataset* File = (GDALDataset*)GDALOpen(FileName.c_str(), GA_ReadOnly);
	//						File->RasterIO(GF_Read, 0, StartY, n_X, PerStep, &vector_inn[n_X * PerStep * n_B * y], n_X, PerStep, type_VI, n_B, nullptr, 0, 0, 0);
	//						GDALClose(File);
	//					}
	//					for (int i = 0; i < n_X * PerStep * n_B * n_Years; i++)
	//						vector_in[i] = float(vector_inn[i]) / 10000;
	//					delete[] vector_inn;
	//					break;
	//				}
	//				default: break;
	//				}
	//				time(&t[1]);
	//				Time[1] += t[1] - t[0];
	//				t[0] = t[1];

	//				int i = 2 + rl % n_Device;
	//				MPI_Send(&rl, 1, MPI_INT, i, 1000000 + i * 100 + rl / n_Device, MPI_COMM_WORLD);//序号+来源*2+目的*2+序列*2
	//				if (rl == ReadLoop - 1)
	//				{
	//					MPI_Send(&PerStep, 1, MPI_INT, i, 3000000 + i * 100 + rl / n_Device, MPI_COMM_WORLD);//序号+来源*2+目的*2+序列*2		
	//				}
	//				MPI_Send(vector_in, n_X * PerStep * n_B * n_Years, MPI_FLOAT, i, 2000000 + i * 100 + rl / n_Device, MPI_COMM_WORLD);//序号+来源*2+目的*2+序列*2

	//				delete[] vector_in;
	//				time(&t[1]);
	//				Time[2] += t[1] - t[0];
	//				t[0] = t[1];
	//			}
	//			cout <<"Read " << Time[0] << " " << Time[1] << " " << Time[2] << " " << Time[3] << " " << Time[4 ] << endl;
	//		}

	//		if (myid > 1 && myid < n_Device + 2)
	//		{
	//			int device = myid - 2;
	//			for (int l = 0; l <= (ReadLoop / n_Device); l++)
	//			{
	//				if (l == (ReadLoop / n_Device) && myid - 1 > (ReadLoop % n_Device))
	//					break;

	//				int rl;
	//				MPI_Recv(&rl, 1, MPI_INT, 0, 1000000 + myid * 100 + l, MPI_COMM_WORLD, &status);
	//				if (rl == ReadLoop - 1)
	//				{
	//					MPI_Recv(&PerStep, 1, MPI_INT, 0, 3000000 + myid * 100 + l, MPI_COMM_WORLD, &status);
	//				}
	//				vector_in = new float[n_X * PerStep * n_B * n_Years];
	//				MPI_Recv(vector_in, n_X * PerStep * n_B * n_Years, MPI_FLOAT, 0, 2000000 + myid * 100 + l, MPI_COMM_WORLD, &status);
	//				time(&t[1]);
	//				Time[2] += t[1] - t[0];
	//				t[0] = t[1];

	//				int StaOrEnd = 0;
	//				int n_Y_d_vector_out = PerStep - 2 * win;
	//				if (rl == 0)
	//				{
	//					StaOrEnd = 1;
	//					n_Y_d_vector_out = PerStep - win;
	//				}
	//				if (rl == ReadLoop - 1)
	//				{
	//					StaOrEnd = 2;
	//					n_Y_d_vector_out = PerStep - win;
	//				}
	//				float* vector_out = new float[n_X * n_Y_d_vector_out * n_B * n_Years];

	//				newSTSG(win, sampcorr, mm, dd, tt, ss, n_Years, n_X, PerStep, n_B, device, StaOrEnd, n_Y_d_vector_out, vector_in, vector_out/*, N[nn]*/);//m2, d2,
	//				time(&t[1]);
	//				Time[3] += t[1] - t[0];
	//				t[0] = t[1];

	//				if (rl == ReadLoop - 1)
	//				{
	//					MPI_Send(&PerStep, 1, MPI_INT, 1, 5000100 + myid * 10000 + l, MPI_COMM_WORLD);//序号+来源*2+目的*2+序列*2	
	//				}

	//				MPI_Send(vector_out, n_X * n_Y_d_vector_out * n_B * n_Years, MPI_FLOAT, 1, 4000100 + myid * 10000 + l, MPI_COMM_WORLD);//序号+来源*2+目的*2+序列*2
	//				time(&t[1]);
	//				Time[2] += t[1] - t[0];
	//				t[0] = t[1];
	//				delete[] vector_in;
	//				delete[] vector_out;
	//			}
	//			cout << "Compute "<<myid<<" " << Time[0] << " " << Time[1] << " " << Time[2] << " " << Time[3] << " " << Time[4] << endl;
	//		}

	//		if (myid == 1)
	//		{
	//			string re_VI_path_1 = "../EVI2_newSTSG_" + to_string(mm) + "_" + to_string(dd) + ".tif";//N[nn]
	//			GDALDataset* File = pDriver->Create(re_VI_path_1.c_str(), n_X, n_Y, n_B * n_Years, GDT_Float32, ppszOptions);
	//			time(&t[1]);
	//			Time[4] += t[1] - t[0];
	//			t[0] = t[1];

	//			for (int rl = 0, StartY = 0 - win; rl < ReadLoop; StartY = (rl + 1) * (PerStep - 2 * win), rl++)
	//			{
	//				int i = 2 + rl % n_Device;

	//				if (rl == ReadLoop - 1)
	//				{
	//					MPI_Recv(&PerStep, 1, MPI_INT, i, 5000100 + i * 10000 + rl / n_Device, MPI_COMM_WORLD, &status);
	//				}
	//				int n_Y_d_vector_out = PerStep - 2 * win;
	//				if (rl == 0)
	//					n_Y_d_vector_out = PerStep - win;
	//				if (rl == ReadLoop - 1)
	//					n_Y_d_vector_out = PerStep - win;
	//				float* vector_out = new float[n_X * n_Y_d_vector_out * n_B * n_Years];
	//				MPI_Recv(vector_out, n_X * PerStep * n_B * n_Years, MPI_FLOAT, i, 4000100 + i * 10000 + rl / n_Device, MPI_COMM_WORLD, &status);
	//				time(&t[1]);
	//				Time[2] += t[1] - t[0];
	//				t[0] = t[1];

	//				File->RasterIO(GF_Write, 0, StartY + win, n_X, n_Y_d_vector_out, vector_out, n_X, n_Y_d_vector_out, GDT_Float32, n_B * n_Years, 0, 0, 0, 0);
	//				cout << "ReadLoop " << rl << endl;
	//				time(&t[1]);
	//				Time[4] += t[1] - t[0];
	//				t[0] = t[1];

	//				delete[] vector_out;
	//			}
	//			GDALClose(File);

	//			cout << "Write " <<Time[0] << " " << Time[1] << " " << Time[2] << " " << Time[3] << " " << Time[4] << endl;

	//			time(&T[1]);
	//			cout << "All " << T[1] - T[0] << endl;
	//		}
	//	}
	//}
	//MPI_Finalize();

	//******************************************************************
	//Test the precision
	//double N[7] = { 0.5,1,1.5,2,2.5,3,5 };
	//for (int nn = 4; nn < 7; nn++)
	//{
	for (int mm = 2; mm < 3; mm++)
	{
		for (int dd = 7; dd < 8; dd++)
		{
			//int mm = 0, dd = 0;

			int n_Years, n_X, n_Y, n_B, Loop;
			int XStep = 500, n_nump = numprocs - 1;
			if (myid == 0)
			{
				n_Years = sizeof(Years) / sizeof(Years[0]);

				string FileName = VI_path + to_string(Years[0]);
				GDALDataset* File = (GDALDataset*)GDALOpen(FileName.c_str(), GA_ReadOnly);
				n_X = File->GetRasterXSize();
				n_Y = File->GetRasterYSize();
				n_B = File->GetRasterCount();
				GDALClose(File);

				Loop = n_X / XStep + 1;
			}
			MPI_Bcast(&n_Years, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&n_X, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&n_Y, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&n_B, 1, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Bcast(&Loop, 1, MPI_INT, 0, MPI_COMM_WORLD);

			if (myid == 0)
			{
				for (int X_i = 0; X_i < Loop; X_i++)
				{
					if (X_i == Loop - 1)
						XStep = n_X - X_i * XStep;

					string re_VI_path_1 = "../EVI2_newSTSG_" + to_string(mm) + "_" + to_string(dd) + ".tif";//  N[nn]
					GDALDataset* File = (GDALDataset*)GDALOpen(re_VI_path_1.c_str(), GA_ReadOnly);
					GDALDataType type_VI = File->GetRasterBand(1)->GetRasterDataType();
					float* img_VI = new float[XStep * n_Y * n_B * n_Years];
					File->RasterIO(GF_Read, X_i * XStep, 0, XStep, n_Y, img_VI, XStep, n_Y, type_VI, n_B * n_Years, nullptr, 0, 0, 0);
					GDALClose(File);

					string FileName = "../EVI2_Reference.tif";
					File = (GDALDataset*)GDALOpen(FileName.c_str(), GA_ReadOnly);
					short* VI_True_buff = new short[XStep * n_Y * n_B];
					File->RasterIO(GF_Read, X_i * XStep, 0, XStep, n_Y, VI_True_buff, XStep, n_Y, GDT_Int16, n_B, nullptr, 0, 0, 0);
					GDALClose(File);

					int i = 1 + X_i % n_nump;
					if (X_i == Loop - 1)
						MPI_Send(&XStep, 1, MPI_INT, i, 300000000 + i * 10000 + X_i / n_nump * 100 + mm * 10 + dd, MPI_COMM_WORLD);//序号+来源*2+目的*2+序列*2	

					MPI_Send(img_VI, XStep * n_Y * n_B * n_Years, MPI_FLOAT, i, 100000000 + i * 10000 + X_i / n_nump * 100 + mm * 10 + dd, MPI_COMM_WORLD);//序号+来源*2+目的*2+序列*2
					MPI_Send(VI_True_buff, XStep * n_Y * n_B, MPI_SHORT, i, 200000000 + i * 10000 + X_i / n_nump * 100 + mm * 10 + dd, MPI_COMM_WORLD);//序号+来源*2+目的*2+序列*2

					delete[] img_VI;
					delete[] VI_True_buff;
				}
			}

			if (myid > 0 && myid < numprocs)
			{
				for (int l = 0; l <= (Loop / n_nump); l++)
				{
					if (l == (Loop / n_nump) && myid > (Loop % n_nump))
						break;

					if (l == (Loop / n_nump) && myid == (Loop % n_nump))
						MPI_Recv(&XStep, 1, MPI_INT, 0, 300000000 + myid * 10000 + l * 100 + mm * 10 + dd, MPI_COMM_WORLD, &status);

					float* img_VI = new float[XStep * n_Y * n_B * n_Years];
					MPI_Recv(img_VI, XStep * n_Y * n_B * n_Years, MPI_FLOAT, 0, 100000000 + myid * 10000 + l * 100 + mm * 10 + dd, MPI_COMM_WORLD, &status);
					short* VI_True_buff = new short[XStep * n_Y * n_B];
					MPI_Recv(VI_True_buff, XStep * n_Y * n_B, MPI_SHORT, 0, 200000000 + myid * 10000 + l * 100 + mm * 10 + dd, MPI_COMM_WORLD, &status);

					double test_true[14]{ 0 };
					for (int i = 0; i < XStep; i++)
					{
						for (int j = 0; j < n_Y; j++)
						{
							for (int k = 0; k < n_B; k++)
							{
								double mean = 0;
								int count = 0;
								for (int y = 0; y < n_Years; y++)
								{
									if (!isnan(img_VI[i + j * XStep + k * XStep * n_Y + y * XStep * n_Y * n_B]) && img_VI[i + j * XStep + k * XStep * n_Y + y * XStep * n_Y * n_B] > -0.2 && img_VI[i + j * XStep + k * XStep * n_Y + y * XStep * n_Y * n_B] <1 &&
										VI_True_buff[i + j * XStep + k * XStep * n_Y] > -2000 && VI_True_buff[i + j * XStep + k * XStep * n_Y] < 10000)
									{
										float x = abs(img_VI[i + j * XStep + k * XStep * n_Y + y * XStep * n_Y * n_B] - (float(VI_True_buff[i + j * XStep + k * XStep * n_Y]) / 10000));
										test_true[11] += x;
										test_true[12] += x * x;
										mean += img_VI[i + j * XStep + k * XStep * n_Y + y * XStep * n_Y * n_B];
										count++;

										switch (int(x * 10))
										{
										case 0:
											test_true[0] = test_true[0] + 1;
											break;
										case 1:
											test_true[1] = test_true[1] + 1;
											break;
										case 2:
											test_true[2] = test_true[2] + 1;
											break;
										case 3:
											test_true[3] = test_true[3] + 1;
											break;
										case 4:
											test_true[4] = test_true[4] + 1;
											break;
										case 5:
											test_true[5] = test_true[5] + 1;
											break;
										case 6:
											test_true[6] = test_true[6] + 1;
											break;
										case 7:
											test_true[7] = test_true[7] + 1;
											break;
										case 8:
											test_true[8] = test_true[8] + 1;
											break;
										default:
											test_true[9] = test_true[9] + 1;
											break;
										}
									}
									else
										test_true[10] = test_true[10] + 1;
								}
								mean = mean / count;
								float x = mean - (float(VI_True_buff[i + j * XStep + k * XStep * n_Y]) / 10000);
								if (!isnan(x))
									test_true[13] += x * x;
							}
						}
					}
					MPI_Send(test_true, 14, MPI_DOUBLE, 0, 400000000 + myid * 1000000 + l * 100 + mm * 10 + dd, MPI_COMM_WORLD);//序号+来源*2+目的*2+序列*2	
				}
			}

			if (myid == 0)
			{
				double MAE = 0;
				double RMSE = 0;
				double R2 = 0;
				for (int X_i = 0; X_i < Loop; X_i++)
				{
					int i = 1 + X_i % n_nump;
					double test_true[14]{ 0 };
					MPI_Recv(test_true, 14, MPI_DOUBLE, i, 400000000 + i * 1000000 + X_i / n_nump * 100 + mm * 10 + dd, MPI_COMM_WORLD, &status);

					cout << X_i << endl;

					MAE += test_true[11] / (n_Y * n_B * n_Years);
					RMSE += test_true[12] / (n_Y * n_B * n_Years);
					R2 += test_true[13];
				}
				MAE = MAE / n_X;
				RMSE = RMSE / n_X;
				R2 = 1 - RMSE * n_X * n_Y * n_B / R2;
				RMSE = sqrt(RMSE);
				cout << mm << "," << dd << " " << MAE << " " << RMSE << " " << R2 << endl;
			}
		}
	}

	MPI_Finalize();
	//******************************************************************

	system("pause");
	return 0;
}

