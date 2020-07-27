/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <stdio.h>

__global__
void max_reduce(const float* const d_in, float* d_out, int inputSize){
    // This function write the maximum value of d_logLuminance into the variable max_logLum. At present, it is    
    // run 1-dimensionally.

    //shared memory declaration
    extern __shared__ float s_in[];
    
    //Thread ids
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
        //make sure we're within the image
    if(idx >= inputSize){
        return;
    }
                       
    //Copy all values of the current thread into shared memory
    s_in[tid] = d_in[idx];
    __syncthreads(); //make sure that thread block is fully loaded into shared memory
                                     
    //Run parallel maximum    
    for(int s = blockDim.x/2; s > 0; s >>= 1){
        if(tid < s){
            s_in[tid] = max(s_in[tid], s_in[tid+s]);
        }
    }
                                          
    __syncthreads();
    
    //copy maxima to output array
    if(tid == 0){
        d_out[blockIdx.x] = s_in[tid];
    }
}

__global__
void min_reduce(const float* const d_in, float* d_out, int inputSize){
    
    //declare shared memory
    extern __shared__ float s_in[];

    //Thread ids
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    //Make sure that threads remain in bounds of input
    if(idx >= inputSize){
        return;
    }

    //copy input array to shared memory for faster execution
    s_in[tid] = d_in[idx];
    __syncthreads();

    //minimum
    for(int s = blockDim.x/2; s > 0; s >>= 1){
        if(tid < s){
            s_in[tid] = min(s_in[tid], s_in[tid+s]);
        }
    }

    __syncthreads();
    
    //copy minima to output array
    if(tid==0){
        d_out[blockIdx.x] = s_in[tid];
    }
}

void parallel_reduce(const float* const d_in, float &address, const size_t numRows, const size_t numCols, int op){
    
    /* Wrapper program for parallel reduction, allows both maximum and minimum
        
        *  d_in : The initial input array
        *  address: memory where the final reduced value is to be stored
        *  numRows: number of rows in input image
        *  numCols: number of columns in input image
        *  op: The operation to run --  0) Maximum
                                        1) Minimum                              */
    
    //declare number of threads per block
    int blockSize = 1024;
    int imgSize = numRows*numCols;
   
    //need a temporary pointer as to not overwrite initial input array when looping
    float *d_temp;
    checkCudaErrors(cudaMalloc((void **)&d_temp, sizeof(float)*imgSize));
    checkCudaErrors(cudaMemcpy(d_temp, d_in, sizeof(float)*imgSize, cudaMemcpyDeviceToDevice));
    
    float *d_out;

    int gridSize = imgSize;
    int inputSize = imgSize;

    //Begin reduction loop
    do{
        //Initialize gridSize of next reduction
        gridSize = floor(gridSize/blockSize)+1;
        
        //Allocate output array based on new gridSize
        checkCudaErrors(cudaMalloc((void **)&d_out, sizeof(float)*gridSize));
        
        //Launch reduce kernel
        if(op == 0){
            max_reduce<<<gridSize, blockSize, sizeof(float)*blockSize>>>(d_temp, d_out, inputSize);
        } else {
            min_reduce<<<gridSize, blockSize, sizeof(float)*blockSize>>>(d_temp, d_out, inputSize);
        }
        //Free and reallocate variables
        checkCudaErrors(cudaFree(d_temp));
        checkCudaErrors(cudaMalloc((void **)&d_temp, sizeof(float)*gridSize));
        checkCudaErrors(cudaMemcpy(d_temp, d_out, sizeof(float)*gridSize, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaFree(d_out));
        
        //Update Size for next iteration's input
        inputSize = gridSize;

    }while(gridSize > 1);
    
    //copy final value to host memory and free remaining device variables
    checkCudaErrors(cudaMemcpy(&address, d_temp, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_temp));
}

__global__
void histogram_atomics(const float* const d_logLuminance, unsigned int* d_hist, float min_logLum, float max_logLum, 
        const size_t numRows, const size_t numCols, const size_t numBins){
    
    //Naively generate histogram from input values using simple atomic addition writing to global memory
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= numRows*numCols){
        return;
    }

    float lumRange = max_logLum - min_logLum;
    int bin = min(static_cast<int>(numBins-1), static_cast<int>(floor((d_logLuminance[idx] - min_logLum) / lumRange * numBins))); 

    atomicAdd(&d_hist[bin], 1);
}

__global__
void histogram_local(const float* const d_logLuminance, unsigned int* d_hist, float min_logLum, float max_logLum, 
        const size_t numRows, const size_t numCols, const size_t numBins, int numPerThread){

    //In this histogram implementation, each thread is responsible for incrementing a local histogram. These histograms
    // are then combined at the end. Due to the large numer of bins we are dealing with, this implementation is quite inefficient.
    
    
    int idx = (blockIdx.x*blockDim.x + threadIdx.x)*numPerThread;

    //Initialize local histogram
    unsigned int local_hist[1024];
    for(int i = 0; i < numBins; i++){
        local_hist[i] = 0;
    }

    //fill local histogram
    for(int i = 0; i < numPerThread; i++){
        if(idx + i < numRows*numCols){
            float lumRange = max_logLum - min_logLum;
            int bin = min(static_cast<int>(numBins-1), static_cast<int>(floor((d_logLuminance[idx+i] - min_logLum) / lumRange * numBins))); 
            local_hist[bin] += 1; 
        }
    }

    for(int i = 0; i < numBins; i++){
        atomicAdd(&d_hist[i], local_hist[i]);
    }
}

__global__
void cdf_scan(unsigned int* d_in, unsigned int* d_out, const size_t numBins){

    //This function generates a cdf from an input histogram. It assumes that the number of bins in the histogram will fit on a single block.
    

    //External memory is declared as 2*sizeof(type)*numBins so that we can have a double buffer in shared memory
    extern __shared__ unsigned int s_cdf[];

    int tid = threadIdx.x;
    
    if(tid >= numBins){
        return;
    }
    
    //Initialize double buffer in shared memory for scan operation
    //Additionally shifts each thread index 1 to the right for exclusive scanning
    int reg1 = 0;
    int reg2 = 1;
    if(tid > 0){
        s_cdf[reg1*numBins + tid] = d_in[tid-1];
        s_cdf[reg2*numBins + tid] = d_in[tid-1];
    } else {
        s_cdf[reg1*numBins + tid] = 0;
        s_cdf[reg2*numBins + tid] = 0;
    }

    __syncthreads();
    
    //Use double buffer to quickly do each step of addition
    for(int step = 1; step < numBins; step <<= 1){
        
        //swap registers
        reg1 = 1 - reg1;
        reg2 = 1 - reg2;

        if(tid >= step){
            s_cdf[reg1*numBins+tid] = s_cdf[reg2*numBins+tid] + s_cdf[reg2*numBins+tid-step];
        } else {
            s_cdf[reg1*numBins+tid] = s_cdf[reg2*numBins+tid];
        }   
        __syncthreads();
    }
    
    //assign from shared to device memory
    d_out[tid] = s_cdf[reg1*numBins + tid];
}

__global__
void blelloch_scan(unsigned int* d_in, unsigned int* d_out, const size_t numBins){
    //This particular scan implementation is taken from Nvidia's "GPU Gems 3"

    extern __shared__ unsigned int s_cdf[];

    int tid = threadIdx.x;
    if(tid >= numBins){
        return;
    }

    int offset = 1;

    //Initialize shared memory array
    s_cdf[2*tid] = d_in[2*tid];
    s_cdf[2*tid+1] = d_in[2*tid+1];
    
    //Reduce Tree Inplace for Upsweep
    for(int d = numBins>>1; d > 0; d >>= 1){
        
        __syncthreads();
        
        if(tid < d){
            int i1 = offset*(2*tid+1) - 1;
            int i2 = offset*(2*tid+2) - 1;
            s_cdf[i2] += s_cdf[i1];
        
        }
        offset <<= 1;
    }

    //Clear last element for exclusive scan
    if(tid == 0){ s_cdf[numBins-1] = 0; }

    //Downsweep Scan
    for(int d = 1; d < numBins; d <<= 1){
        
        offset >>= 1;
        __syncthreads();
        
        if(tid < d){
            int i1 = offset*(2*tid+1) - 1;
            int i2 = offset*(2*tid+2) - 1;
            unsigned int temp = s_cdf[i1];
            s_cdf[i1] = s_cdf[i2];
            s_cdf[i2] += temp;
        }

    }
    __syncthreads();

    //Write to output
    d_out[2*tid] = s_cdf[2*tid];
    d_out[2*tid+1] = s_cdf[2*tid+1];

}

void parallel_cdf(const float* const d_logLuminance, unsigned int* const d_cdf, float min_logLum, float max_logLum, 
        const size_t numRows, const size_t numCols, const size_t numBins){
    
    /* This function generates the cdf by first creating the histogram and then scanning it*/ 
    int blockSize = 1024;
    int imgSize = numRows * numCols;
    int gridSize = floor(imgSize/blockSize)+1;
    
    //allocate memory for histogram
    unsigned int *d_hist;
    checkCudaErrors(cudaMalloc((void **)&d_hist, sizeof(unsigned int)*numBins)); //uint[numBins]
    checkCudaErrors(cudaMemset(d_hist, 0, sizeof(unsigned int)*numBins));
    
    //generate histogram using the naive atomics method
    histogram_atomics<<<gridSize, blockSize>>>(d_logLuminance, d_hist, min_logLum, max_logLum, numRows, numCols, numBins);
    
    /* This is an implementation of the histogram where each thread is resposible for some number of elements binned in local memory. These are then combined atomically to global memory at the end.
       Due to the large number of bins in this particular application, this method is substantially less efficient than the naive histogram method.
    int numPerThread = 256;
    histogram_local<<<floor(gridSize/numPerThread)+1, blockSize>>>(d_logLuminance, d_hist, min_logLum, max_logLum, 
                                                                        numRows, numCols, numBins, numPerThread); */
    cdf_scan<<<1, numBins, sizeof(unsigned int)*numBins*2>>>(d_hist, d_cdf, numBins);
    //blelloch_scan<<<1, numBins/2, sizeof(unsigned int)*numBins>>>(d_hist, d_cdf, numBins);
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{

    //Find maximum and minimum luminosity values with parallel reduction
    parallel_reduce(d_logLuminance, max_logLum, numRows, numCols, 0);
    parallel_reduce(d_logLuminance, min_logLum, numRows, numCols, 1);

    //generate histogram -> cdf
    parallel_cdf(d_logLuminance, d_cdf, min_logLum, max_logLum, numRows, numCols, numBins);    

}


