// MP 2: Due Sunday, Dec 16, 2012 at 11:59 p.m. PST
#include    <wb.h>

#define TILE_WIDTH 32

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return -1;                                     \
        }                                                  \
    } while(0)

// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
			       int numARows, int numAColumns,
			       int numBRows, int numBColumns,
			       int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    __shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;    
  
  	float Cvalue = 0;
  	for (int tileNumber = 0; tileNumber <= numAColumns/TILE_WIDTH; ++tileNumber) {
      	int tileStart = tileNumber * TILE_WIDTH;
      
        int actualWidth = numAColumns - tileStart; 
      	//if it is tile near the border - real tile width count could be less then TILE_WIDTH - crop in this case
      	if (actualWidth > TILE_WIDTH) actualWidth = TILE_WIDTH;		
 
      	if (threadIdx.x < actualWidth && row < numARows) {      
          	s_A[threadIdx.y][threadIdx.x] = A[row*numAColumns + tileStart + threadIdx.x]; 	        
        }
      
        if (col < numBColumns && threadIdx.y < actualWidth) {      
      		s_B[threadIdx.y][threadIdx.x] = B[(tileStart+threadIdx.y)*numBColumns+col];	        
        }
      	__syncthreads();
		   
          for (int e = 0; e < actualWidth; ++e) {
            float a = s_A[threadIdx.y][e];
            float b = s_B[e][threadIdx.x];
            Cvalue += a * b;
          }
  	}
    if(row >= numCRows || col >= numCColumns) return;
  	C[row * numCColumns + col] = Cvalue;
  	__syncthreads();
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
    
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    wbCheck(cudaMalloc((void**)&deviceA, numARows * numAColumns * sizeof(float)));  
    wbCheck(cudaMalloc((void**)&deviceB, numBRows * numBColumns * sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceC, numCRows * numCColumns * sizeof(float)));
  
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
      
    wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceC, hostC, numCRows * numCColumns * sizeof(float), cudaMemcpyHostToDevice));
    
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here  
    dim3 grid(ceil((float)numCColumns/TILE_WIDTH), ceil((float)numCRows/TILE_WIDTH), 1);
    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
  
      
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiply<<<grid, block>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    cudaThreadSynchronize();
    wbCheck(cudaGetLastError());
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize(); 
  
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    wbCheck(cudaFree(deviceA));
    wbCheck(cudaFree(deviceB));
    wbCheck(cudaFree(deviceC));

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

