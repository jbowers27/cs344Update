//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

//We are using a 2bit per loop radix sort
#define RADIX_SIZE 4

__global__
void radix2bit_blockCount(unsigned int* d_inputVals, unsigned int* d_mask, unsigned int* d_blockSum, const size_t numElems, int numPerThread, int offset){

    // This algorithm generates a histogram describing the input based on trailing bit-values 00, 01, 10, 11
    int idx = (blockIdx.x*blockDim.x + threadIdx.x)*numPerThread;
    int bDim = blockDim.x*numPerThread;
    //Declare local histogram and initialize values to 0
    unsigned int local_hist[RADIX_SIZE];
    for(int i = 0; i < RADIX_SIZE; i++){
        local_hist[i] = 0;
    }
   
    //Serially add 1 to each bin 
    for(int i = 0; i < numPerThread; i++){
        if(idx + i < numElems){
            int bin = (d_inputVals[idx+i]>>offset) & (RADIX_SIZE - 1);
            local_hist[bin] += 1;
            d_mask[bin*bDim*gridDim.x+idx+i] = 1;
        }
    }
    
    for(int b = 0; b < RADIX_SIZE; b++){
        atomicAdd(&d_blockSum[b*gridDim.x + blockIdx.x],local_hist[b]);
    } 
}

__global__
void blelloch_scan(unsigned int* d_blockSum, const size_t numBins){
    
    //This particular scan implementation is adapted from Nvidia's "GPU Gems 3" for use with non-power-of-2 length arrays
    extern __shared__ unsigned int s_sum[];

    int tid = threadIdx.x;
    int bDim = blockDim.x*2;

    //Initialize shared memory array
    // Threads beyond the scope of d_blockSum -> 0
    if(2*tid >= numBins){
        s_sum[2*tid] = 0;
    } else {
        s_sum[2*tid] = d_blockSum[2*tid];
    }
    
    if(2*tid+1 >= numBins){
        s_sum[2*tid+1] = 0;
    } else {
        s_sum[2*tid+1] = d_blockSum[2*tid+1]; 
    }

    //Reduce Tree Inplace for Upsweep
    int offset = 1;
    for(int d = bDim>>1; d > 0; d >>= 1){
        
        __syncthreads();
        
        if(tid < d){
            int i1 = offset*(2*tid+1) - 1;
            int i2 = offset*(2*tid+2) - 1;
            s_sum[i2] += s_sum[i1];
        
        }
        offset <<= 1;
    }

    //Clear last element for exclusive scan
    if(tid == 0){ s_sum[bDim-1] = 0; }

    //Downsweep Scan
    for(int d = 1; d < bDim; d <<= 1){
        
        offset >>= 1;
        __syncthreads();
        
        if(tid < d){
            int i1 = offset*(2*tid+1) - 1;
            int i2 = offset*(2*tid+2) - 1;
            unsigned int temp = s_sum[i1];
            s_sum[i1] = s_sum[i2];
            s_sum[i2] += temp;
        }

    }
    __syncthreads();    

    //Write only non-zero sums to output
    if(2*tid < numBins){   
        d_blockSum[2*tid] = s_sum[2*tid];
    }

    if(2*tid+1 < numBins){
        d_blockSum[2*tid+1] = s_sum[2*tid+1];
    }
}

__global__
void value_shuffle(unsigned int* d_mask, unsigned int* d_blockSum, int numElems, unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos){
    
    //First we need to scan d_mask to get the relative offests of each value
    extern __shared__ unsigned int s_mask[];

    //Load into shared memory
    int tid = threadIdx.x;
    int idx = blockIdx.x*blockDim.x*2 + 2*threadIdx.x;
    int bDim = blockDim.x*2;

    for(int bin = 0; bin < RADIX_SIZE; bin++){
        s_mask[bin*bDim + 2*tid] = d_mask[bin*bDim*gridDim.x + idx];
        s_mask[bin*bDim + 2*tid+1] = d_mask[bin*bDim*gridDim.x + idx + 1];
    }

    __syncthreads();

    //Now Blelloch Scan each bin of the shared memory array
    for(int bin = 0; bin < RADIX_SIZE; bin++){
    
        //Reduce Tree Inplace for Upsweep
        int blockOff = bin*bDim;
        int offset = 1;
        for(int d = bDim>>1; d > 0; d >>= 1){
        
            __syncthreads();
        
            if(tid < d){
                int i1 = offset*(2*tid+1) - 1;
                int i2 = offset*(2*tid+2) - 1;
                s_mask[blockOff + i2] += s_mask[blockOff + i1];
        
            }
            offset <<= 1;
        }

        //Clear last element for exclusive scan
        if(tid == 0){ s_mask[blockOff+bDim-1] = 0; }

        //Downsweep Scan
        for(int d = 1; d < bDim; d <<= 1){
            
            offset >>= 1;
            __syncthreads();
            
            if(tid < d){
                int i1 = offset*(2*tid+1) - 1;
                int i2 = offset*(2*tid+2) - 1;
                unsigned int temp = s_mask[blockOff + i1];
                s_mask[blockOff + i1] = s_mask[blockOff + i2];
                s_mask[blockOff + i2] += temp;
            }
        }
 
        //end scanning of s_mask now we have all the information we need to find absolute positions in the input array
        __syncthreads();
       
        if(d_mask[bin*bDim*gridDim.x + idx] && idx < numElems){
            int ido = s_mask[bin*bDim + 2*tid] + d_blockSum[bin*gridDim.x + blockIdx.x];
            d_outputVals[ido] = d_inputVals[idx];
            d_outputPos[ido] = d_inputPos[idx];
        }
        if(d_mask[bin*bDim*gridDim.x + idx+1] && idx+1 < numElems){
            int ido = s_mask[bin*bDim + 2*tid+1] + d_blockSum[bin*gridDim.x + blockIdx.x];
            d_outputVals[ido] = d_inputVals[idx+1];
            d_outputPos[ido] = d_inputPos[idx+1];
        }
    } 
} 

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
    //CUDA Kernel Call Variables
    int blockSize = 1024;
    int numPerThread = 16;
    int gridSize = (floor(numElems/blockSize) + 1);
    
    //loop variables
    int offset = 0;
    int swap = 1;
    
    //Initialize GPU device variables for use
    unsigned int *d_blockSum, *d_mask;
    checkCudaErrors(cudaMalloc((void **)&d_blockSum, sizeof(unsigned int)*RADIX_SIZE*gridSize));
    checkCudaErrors(cudaMalloc((void **)&d_mask, sizeof(unsigned int)*RADIX_SIZE*blockSize*gridSize));
        
    for(int i = 0; i < 16; i++){ 
        
        //Zero out device variables at each iteration
        checkCudaErrors(cudaMemset(d_mask, 0, sizeof(unsigned int)*RADIX_SIZE*blockSize*gridSize));
        checkCudaErrors(cudaMemset(d_blockSum, 0, sizeof(unsigned int)*RADIX_SIZE*gridSize));
        
        //if most recent result is in input
        if(swap % 2){
            
            //Calculate relative positions of each digit for each block
            radix2bit_blockCount<<<gridSize, blockSize/numPerThread>>>(d_inputVals, d_mask, d_blockSum, 
                    numElems, numPerThread, offset);            

            //Scan blockSum array
            blelloch_scan<<<1, blockSize/2, sizeof(unsigned int)*blockSize>>>(d_blockSum, gridSize*RADIX_SIZE);
           
             //shuffle values
            value_shuffle<<<gridSize, blockSize/2, sizeof(unsigned int)*blockSize*RADIX_SIZE>>>(d_mask, d_blockSum, numElems, 
                    d_inputVals, d_inputPos, d_outputVals, d_outputPos);
        
        //if most recent result is in output
        } else {
  
            //Calculate relative positions of each digit for each block
            radix2bit_blockCount<<<gridSize, blockSize/numPerThread>>>(d_outputVals, d_mask, d_blockSum, 
                    numElems, numPerThread, offset);
            
            //Scan blockSum array
            blelloch_scan<<<1, blockSize/2, sizeof(unsigned int)*blockSize>>>(d_blockSum, gridSize*RADIX_SIZE);
            
            //shuffle values
            value_shuffle<<<gridSize, blockSize/2, sizeof(unsigned int)*blockSize*RADIX_SIZE>>>(d_mask, d_blockSum, numElems, 
                    d_outputVals, d_outputPos, d_inputVals, d_inputPos);
        } 
        
        //increment loop variables
        offset += 2;
        swap += 1;
    }
    
    //16 Iterations means our final result is in the d_input* variables
    checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, sizeof(unsigned int)*numElems, cudaMemcpyDeviceToDevice));

    //free device memory
    checkCudaErrors(cudaFree(d_blockSum));
    checkCudaErrors(cudaFree(d_mask));     
}
