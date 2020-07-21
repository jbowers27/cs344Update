#include <stdio.h>
#include "gputimer.h"

//We have made a number of edits to every function within this program to enable us to run all test cases from Lesson 2(3) Lecture 38 simultaneously.

//Reduced to printing only 25 elements for aesthetic output
void print_array(int *array, int size)
{
    printf("{ ");
    for (int i = 0; i < 25; i++)  { printf("%d ", array[i]); }
    printf("...}\n");
}

//Now takes a pointer to ARRAY_SIZE to let us run main() as a for-loop
__global__ void increment_naive(int *g, int *ARRAY_SIZE)
{
	// which thread is this?
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
    
	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	i = i % *ARRAY_SIZE;  
	g[i] = g[i] + 1;
}

//Now takes a pointer to ARRAY_SIZE to let us run main() as a for-loop
__global__ void increment_atomic(int *g, int *ARRAY_SIZE)
{
	// which thread is this?
	int i = blockIdx.x * blockDim.x + threadIdx.x; 

	// each thread to increment consecutive elements, wrapping at ARRAY_SIZE
	i = i % *ARRAY_SIZE;  
	atomicAdd(& g[i], 1);
}

int main(int argc,char **argv)
{   
    GpuTimer timer;
    
    /////////////////////////////////////////////////////////////////////////
    //                                                                     //
    //  We create arrays containing all of the desired test cases:         //
    //  * 10^6 threads incrementing 10^6 elements                          //
    //  * 10^6 threads atomically incrementing 10^6 elements               //
    //  * 10^6 threads incrementing 100 elements                           //
    //  * 10^6 threads atomically incrementing 100 elements                //
    //  * 10^7 threads atomically incrementing 100 elements                //
    //                                                                     //
    //  Increment method is determined by use[]: 0 for naive, 1 for atomic //
    //                                                                     //
    /////////////////////////////////////////////////////////////////////////
    int NUM_THREADS [5] = {1000000, 1000000, 1000000, 1000000, 10000000};
    int BLOCK_WIDTH [5] = {1000, 1000, 1000, 1000, 1000};
    int h_ARRAY_SIZE[5] = {1000000, 1000000, 100, 100, 100};
    int use[5] = {0, 1, 0, 1, 1};
    
    for(int i = 0; i < 5; i++){
   
        //Let's explicitly create a device version of our ARRAY_SIZE variable for now. This isn't strictly necessary, 
        //given that the nvcc compiler would implicitly marshall the value of h_ARRAY_SIZE to the device if called directly.
        //But this code is hidden and we may as well write it explicitly for practice.
        int *d_ARRAY_SIZE;
        cudaMalloc((void **) &d_ARRAY_SIZE, sizeof(int));
        cudaMemcpy(d_ARRAY_SIZE, &h_ARRAY_SIZE[i], sizeof(int), cudaMemcpyHostToDevice);
        printf("%d total threads in %d blocks writing into %d array elements\n",
               NUM_THREADS[i], NUM_THREADS[i] / BLOCK_WIDTH[i], h_ARRAY_SIZE[i]);
        
        // declare and allocate host memory
        int h_array[h_ARRAY_SIZE[i]];
        int ARRAY_BYTES = h_ARRAY_SIZE[i] * sizeof(int);
        
        // declare, allocate, and zero out GPU memory
        int * d_array;
        cudaMalloc((void **) &d_array, ARRAY_BYTES);
        cudaMemset((void *) d_array, 0, ARRAY_BYTES); 

        // Launch the kernel. Increment naively or atomically depending on test case.
        timer.Start();

        if(use[i] == 0){
            increment_naive<<<NUM_THREADS[i]/BLOCK_WIDTH[i], BLOCK_WIDTH[i]>>>(d_array, d_ARRAY_SIZE);
        } else {
            increment_atomic<<<NUM_THREADS[i]/BLOCK_WIDTH[i], BLOCK_WIDTH[i]>>>(d_array, d_ARRAY_SIZE);
        }
        timer.Stop();
        
        // copy back the array of sums from GPU and print
        cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
        print_array(h_array, h_ARRAY_SIZE[i]);
        printf("Time elapsed = %g ms\n\n", timer.Elapsed());
     
        // free GPU memory allocation and exit
        cudaFree(d_array);
        cudaFree(d_ARRAY_SIZE);
    }
    return 0;
}
