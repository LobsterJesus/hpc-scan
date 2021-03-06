#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

// Avoid Visual Studio LNK2019 compiler error
#pragma comment(lib, "OpenCL.lib")

using namespace std;

int main() {

	cl_int err;
	cl_platform_id platforms[8];
	cl_uint numPlatforms;
	cl_device_id device;

	err = clGetPlatformIDs(8, platforms, &numPlatforms);
	if (err != CL_SUCCESS) {
		cerr << "OpenCL platforms not found." << std::endl;
		return 1;
	}

	err = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if (err != CL_SUCCESS) {
		cerr << "Device not found." << std::endl;
		return 1;
	}

	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
	cl_command_queue queue = clCreateCommandQueue(context, device, (cl_command_queue_properties)0, &err);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't create command queue." << std::endl;
		return 1;
	}

	ifstream kernelFS("BlellochScan.cl");
	string kernelSourceString((istreambuf_iterator<char>(kernelFS)), (istreambuf_iterator<char>()));
	const char *kernelSource = &kernelSourceString[0u];

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &err);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't create program object." << std::endl;
		return 1;
	}

	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {

		fprintf(stderr, "Couldn't build program (%d).\n", err);
		return 1;
	}

	cl_kernel scan = clCreateKernel(program, "blellochScan", &err);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't create kernel." << std::endl;
		return 1;
	}
	
	vector<int> scanInput = {1, 2, 3, 4};
	vector<int> scanOutput(scanInput.size(), 0);
	uint32_t scanArraySize = scanInput.size() * sizeof(int);

	cl_mem scanInBuffer =
		clCreateBuffer(context, CL_MEM_READ_ONLY, scanArraySize, NULL, &err);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't create kernel input buffer." << std::endl;
		return 1;
	}

	cl_mem scanOutBuffer =
		clCreateBuffer(context, CL_MEM_READ_WRITE, scanArraySize, NULL, &err);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't create kernel output buffer." << std::endl;
		return 1;
	}
	
	
	err = clEnqueueWriteBuffer(queue, scanInBuffer, CL_TRUE, 0, scanArraySize, scanInput.data(), 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't write image input buffer." << std::endl;
		return 1;
	}

	int numberOfElements = scanInput.size();
	clSetKernelArg(scan, 0, sizeof(cl_mem), (void *)&scanInBuffer);
	clSetKernelArg(scan, 1, sizeof(cl_mem), (void *)&scanOutBuffer);
	clSetKernelArg(scan, 2, sizeof(cl_int), (void *)&numberOfElements);

	size_t globalItemSize = scanInput.size();
	size_t localItemSize = globalItemSize;

	err = clEnqueueNDRangeKernel(queue, scan, 1, 0, &globalItemSize, &localItemSize, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't enqueue command to execute on kernel." << std::endl;
		return 1;
	}

	err = clEnqueueReadBuffer(queue, scanOutBuffer, CL_TRUE, 0, scanArraySize, scanOutput.data(), NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		cerr << "Couldn't enqueue command to read from kernel." << std::endl;
		return 1;
	}

	if (clFinish(queue) != CL_SUCCESS) {
		cerr << "Couldn't finish queue." << std::endl;
		return 1;
	}
	
	printf("output size: %d\n", scanOutput.size());

	for (unsigned int i = 0; i < scanOutput.size(); i++) {
		printf("at %d: %d\n", i, scanOutput[i]);
	}

	return 0;
}