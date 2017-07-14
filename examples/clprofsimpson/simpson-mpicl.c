// This program tests Simpson theorem using OpenCL
 
// MPI Includes
#include <mpi.h>
 
// System includes
#include <stdio.h>
#include <stdlib.h>
 
// OpenCL includes
#include <CL/cl.h>
 
#define MAX_SOURCE_SIZE (0x100000)
// Define min macro
#define min(X, Y) ((X) < (Y) ? (X) : (Y))
 
void equal_load(int, int, int, int, int*, int*);
int main(int argc, char** argv)
{
    int myid, nproc;
    int n = 500000, i;
    float sum = 0.L;
    float aa = 0.L;
    float bb = 1.L;
    float time_start, time_end;
    float h = (bb-aa)/(float)n;
    int istart, ifinish;
 
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
 
    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
 
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
 
    // Check starting time
    time_start = MPI_Wtime();
 
 
    // Load the source code containing the kernel
    FILE *fp;
    char fileName[] = "./simpson-mpicl.cl";
    char *source_str;
    size_t source_size;
 
    /* Load the source code containing the kernel*/
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
 
    // Load equally
    equal_load(1, n-1, nproc, myid, &istart, &ifinish);
    printf("(%d / %d) calculate from %d to %d\n", myid+1, nproc, istart, ifinish);
 
    // num elements
    int nelem = ifinish - istart + 1;
 
    // This code executes on the OpenCL host
    // Host data
    float *X=NULL; // Input array
    float *Y=NULL; // Output array
 
    // Compute the size of the data
    size_t datasize=sizeof(float)*nelem;
 
    // Allocate space for input/output data
    X=(float*)malloc(datasize);
    Y=(float*)malloc(datasize);
 
    // Initialize the input data
    for(i=0; i<nelem; i++){
        X[i]= aa + h*(float)(i+istart);
    }
     
    // Use this to check the output of each API call
    cl_int status;
 
    // Retrieve the number of platforms
    cl_uint numPlatforms=0;
    status=clGetPlatformIDs(0, NULL, &numPlatforms);
 
    // Allocate enough space for each platform
    cl_platform_id *platforms=NULL;
    platforms=(cl_platform_id*)malloc(
        numPlatforms*sizeof(cl_platform_id));
 
    // Fill in the platforms
    status=clGetPlatformIDs(numPlatforms, platforms, NULL);
 
    // Retrieve the number of devices
    cl_uint numDevices=0;
    status=clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0,
        NULL, &numDevices);
 
    // Allocate enough space for each device
    cl_device_id *devices;
    devices=(cl_device_id*)malloc(
        numDevices*sizeof(cl_device_id));
 
    // Fill in the devices
    status=clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL,
        numDevices, devices, NULL);
 
    // Create a context and associate it with the devices
    cl_context context;
    context=clCreateContext(NULL, numDevices, devices, NULL,
        NULL, &status);
 
    // Create a command queue and associate it with the device
    cl_command_queue cmdQueue;
    cmdQueue=clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE,
        &status);
 
    // Create a buffer object that will contain the data
    // from the host array A
    cl_mem bufX;
    bufX=clCreateBuffer(context, CL_MEM_READ_ONLY, datasize,
        NULL, &status);
 
    // Create a buffer object that will contain the output data
    // from the host array Y
    cl_mem bufY;
    bufY=clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize,
        NULL, &status);
 
    // Write input array A to the device buffer bufferA
    status=clEnqueueWriteBuffer(cmdQueue, bufX, CL_FALSE,
        0, datasize, X, 0, NULL, NULL);
 
    // Write input array A to the device buffer bufferB
    status=clEnqueueWriteBuffer(cmdQueue, bufY, CL_FALSE,
        0, datasize, Y, 0, NULL, NULL);
 
    // Create a program with source code
    cl_program program=clCreateProgramWithSource(context, 1,
        (const char**)&source_str, NULL, &status);
 
    // Build (compile) the program for the device
    status=clBuildProgram(program, numDevices, devices,
        NULL, NULL, NULL);
 
    // Create the vector addition kernel
    cl_kernel kernel;
    kernel=clCreateKernel(program, "func", &status);
 
    // Associate the input and output buffers with the kernel
    status=clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufX);
    status=clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufY);
 
    // Define an index space (global work size) of work
    // items for execution. A workgroup size (local work size)
    // is not required, but can be used.
    size_t globalWorkSize[1];
 
    // There are 'nelem' work-items
    globalWorkSize[0]=nelem;
 
	cl_event event;
    // Execute the kernel for execution
    status=clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL,
        globalWorkSize, NULL, 0, NULL, &event);

 
    // Read the device output buffer to the host output array
    clEnqueueReadBuffer(cmdQueue, bufY, CL_TRUE, 0,
        datasize, Y, 0, NULL, NULL);
    // Reduce the output
    for(i=0; i<nelem; i++) {
        if ((myid == 0 && i == 0) || (myid == nproc-1 && i == nelem-1)) sum += Y[i];
        else if ((istart + i)%2 == 0) sum += Y[i] * 2;
        else sum += Y[i] * 4;
    }
 
    sum = sum * h / 3.L;
 
 
     
    if(nproc > 1) {
        if(myid > 0) { // Slave
            MPI_Send(&sum, 1, MPI_FLOAT, 0, 42, MPI_COMM_WORLD);
        }
        else { // Master
            float total = sum;
            float tmp;
            for(i=1; i<nproc; i++){
                MPI_Recv(&tmp, 1, MPI_FLOAT, i, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                total += tmp;
            }
            printf("Result = %f\n", total);
        }
    }
    else {
        printf("Result = %f\n", sum);
    }
 
    time_end = MPI_Wtime();
    if(myid==0) {
        float timer = time_end - time_start;
        printf("%14.5g s\n", timer);
    }


	cl_ulong t_start, t_end;
	t_start = t_end = 0;
	double t_time;

	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_end, NULL);
	t_time = t_end - t_start;
	printf("\n[Rank %d] Execution time in milliseconds = %0.3f ms\n", myid+1, (t_time / 1000000.0) );
 
    // Free OpenCL resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufX);
    clReleaseMemObject(bufY);
    clReleaseContext(context);

    // Free host resources
    free(source_str);
    free(platforms);
    free(devices);
    free(X);
    free(Y);
 
    // Finalize the MPI environment.
    MPI_Finalize();
 
    return 0;
}
 
void equal_load(int n1, int n2, int nproc, int myid, int *istart, int *ifinish) {
    int iw1, iw2;
    iw1 = (n2-n1+1)/nproc;
    iw2 = (n2-n1+1)%nproc;
    *istart = myid*iw1 + n1 + min(myid, iw2);
    *ifinish = *istart + iw1 - 1;
    if (iw2 > myid) *ifinish = *ifinish + 1;
    if (n2 < *istart) *ifinish = *istart - 1;
}
