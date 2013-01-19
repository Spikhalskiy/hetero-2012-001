// MP 5 Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}
// Due Tuesday, January 22, 2013 at 11:59 p.m. PST

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

__global__ void scan(float * input, float * blockSum, int len) {
    __shared__ float scan_array[2 * BLOCK_SIZE];
    int index, stride;  
  
    int i = 2 * blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    if (i < len) {
        scan_array[threadIdx.x] = input[i];
    } else {
        scan_array[threadIdx.x] = 0;
    }
    
    int i2 = i + BLOCK_SIZE;
  
    if (i2 < len) {
        scan_array[threadIdx.x + BLOCK_SIZE] = input[i2];
    } else {
        scan_array[threadIdx.x + BLOCK_SIZE] = 0;
    }
  
    //reduction step  
    stride = 1;
    while (stride < 2 * BLOCK_SIZE) {//TODO maybe 2 *
        __syncthreads();
        index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < 2 * BLOCK_SIZE) scan_array[index] += scan_array[index - stride];
        stride *= 2;
    }
    
    if (threadIdx.x == 0) {
      blockSum[blockIdx.x] = scan_array[2 * BLOCK_SIZE - 1];
    }
  
    
    //back - post reduction step
    for (stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();      
        index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < 2 * BLOCK_SIZE) scan_array[index + stride] += scan_array[index];   
    }
    __syncthreads();
  
    if (i < len)
        input[i] = scan_array[threadIdx.x];
    if (i2 < len)
        input[i2] = scan_array[threadIdx.x + BLOCK_SIZE];
  
}

__global__ void post(float* input, float* PartialSum) {
    int dataBlockIndex = blockIdx.x + 1;
    input[dataBlockIndex * 2 * BLOCK_SIZE + threadIdx.x] += PartialSum[dataBlockIndex];
    input[dataBlockIndex * 2 * BLOCK_SIZE + threadIdx.x + BLOCK_SIZE] += PartialSum[dataBlockIndex];
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    int numElements; // number of elements in the list
    float * deviceBlockSum;
    float * hostBlockSum;
    float * devicePartialSum;
    float * hostPartialSum;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    int gridSize = ceil((float)numElements/(2 * BLOCK_SIZE)); 
    dim3 dimGrid(gridSize, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);
    
  
    hostBlockSum = (float*) malloc(gridSize * sizeof(float));
    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceBlockSum, gridSize*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");  
  
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the device
      
    scan<<<dimGrid, dimBlock>>>(deviceInput, deviceBlockSum, numElements);

    wbCheck(cudaDeviceSynchronize());
    wbTime_stop(Compute, "Performing CUDA computation");  
  
    int postGridSize = ceil((float)numElements/(2 * BLOCK_SIZE)) - 1;
    if (postGridSize > 0) {
        hostPartialSum = (float*) malloc(gridSize * sizeof(float));
        wbTime_start(GPU, "Allocating GPU memory.");
        wbCheck(cudaMalloc((void**)&devicePartialSum, gridSize*sizeof(float)));
        wbTime_stop(GPU, "Allocating GPU memory.");  
      
        wbCheck(cudaMemcpy(hostBlockSum, deviceBlockSum, gridSize*sizeof(float), cudaMemcpyDeviceToHost));

        hostPartialSum[0]=0;
        for (int ii = 1; ii<gridSize; ii++) { 
            hostPartialSum[ii] = hostPartialSum[ii-1] + hostBlockSum[ii-1];
        }

        wbCheck(cudaDeviceSynchronize());
        wbCheck(cudaMemcpy(devicePartialSum, hostPartialSum, gridSize*sizeof(float), cudaMemcpyHostToDevice));
      
        dim3 dimGrid(postGridSize, 1, 1);
        dim3 dimBlock(BLOCK_SIZE, 1, 1);
        wbTime_start(Compute, "Performing post CUDA computation");
        post<<<dimGrid, dimBlock>>>(deviceInput, devicePartialSum);
        wbCheck(cudaDeviceSynchronize());
        wbTime_stop(Compute, "Performing post CUDA computation");
    }
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceInput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");  
  
      

    

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceBlockSum);
    if (postGridSize > 0) cudaFree(devicePartialSum);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);
    free(hostBlockSum);
    if (postGridSize > 0) free(hostPartialSum);

    return 0;
}
