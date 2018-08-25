#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#define ALIVE 1
#define DEAD 0

#  define CUDA_SAFE_CALL( call) {                                    \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

int termCheck(int *prev, int *next, int N, int M, int gen);
__global__ void kernel_update(int* t,int* t1,int N,int M);
__global__ void initdat(int *t, int *t1, int N, int M, time_t clock);

int main(int argc, char *argv[]) {

	int N,				/* rows of the grid */
		M,				/* columns of the grid */
		Gens,			/* amount of generations. */
		perGens;		/* checks termination per perGens generations. if perGens zero no termination check */
	int *grid, *grid1;
	int *gpu_grid, *gpu_grid1, *gpu_temp;

	if ( argc != 5) {
		printf("Error! Missing mandatory argument.\n");
		return 1;
	}

	N = atoi(argv[1]); /* Getting rows amount */
	M = atoi(argv[2]); /* Getting columns amount */
	Gens = atoi(argv[3]); /* Getting Gens */
	perGens = atoi(argv[4]);
	if (Gens <= 0 || N < 0 || M < 0 || perGens < 0) {
		printf("Please give positive values for rows/cols and Generations\n");
		return 1;
	}

	int blockSize = 512;
	int numBlocks = (N*M + blockSize - 1) / blockSize;

	grid = (int*)malloc(sizeof(int)*N*M);
	grid1 = (int*)malloc(sizeof(int)*N*M);

	CUDA_SAFE_CALL(cudaMalloc(&gpu_grid, N*M*sizeof(int)));
	CUDA_SAFE_CALL(cudaMalloc(&gpu_grid1, N*M*sizeof(int)));

	/* Initialize random data */
	initdat<<<numBlocks,blockSize>>>(gpu_grid, gpu_grid1, N, M, time(NULL));
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	for (int k = 1; k <= Gens; k++) {

		kernel_update<<<numBlocks,blockSize>>>(gpu_grid,gpu_grid1,N,M);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		if ( perGens ) {
			CUDA_SAFE_CALL(cudaMemcpy(grid, gpu_grid, N*M*sizeof(int), cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL(cudaMemcpy(grid1, gpu_grid1, N*M*sizeof(int), cudaMemcpyDeviceToHost));

			if ( k % perGens == 0) {
				if (termCheck(grid, grid1, N, M, k)) {
					cudaFree(gpu_grid1);
					cudaFree(gpu_grid);
					free(grid);
					free(grid1);
					return 0;
				}
			}
		}
		gpu_temp = gpu_grid;
		gpu_grid = gpu_grid1;
		gpu_grid1 = gpu_temp;
	}
	printf("Reached requested generations %d\n",Gens );

	cudaFree(gpu_grid1);
	cudaFree(gpu_grid);
	free(grid);
	free(grid1);

	return 0;
}

int termCheck(int *prev, int *next, int N, int M, int gen){
	int allDiff = 0;
	int sum = 0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			if (prev[i*M+j] != next[i*M+j]) {
				allDiff = 1;
			}
			sum += next[i*M+j];
		}
	}
	if (!sum) {
		printf("All cells are dead at generation %d\n", gen);
		return 1;
	}
	else if (!allDiff) {
		printf("Generation %d is the same with generation %d\n", gen, gen-1);
		return 1;
	}
	return 0;
}


__global__ void kernel_update(int* t,int* t1,int N,int M){

    int x = blockIdx.x * blockDim.x + threadIdx.x;

    /*update starts*/
    if (0 <= x && x < N*M ){     //if not out of bounds then..
        int i,j,neighbours;
        i = x / M;
        j = x % M;
        if (i+1 > N-1) {
            if (j-1 < 0) {
                /* eimai o bot_left */
				neighbours = t[(i-1)*M+M-1] + t[(i-1)*M+j] + t[(i-1)*M+j+1] + t[i*M+j+1] + t[0*M+j+1] + t[0*M+j] + t[0*M+M-1] + t[i*M+M-1];
			}
            else if (j+1 > M-1) {
                /* eimai o bot_right */
				neighbours = t[(i-1)*M+j-1] + t[(i-1)*M+j] + t[(i-1)*M+0] + t[i*M+0] + t[0*M+0] + t[0*M+j] + t[0*M+j-1] + t[i*M+j-1];
			}
            else{
                /* eimai aplos bot */
				neighbours = t[(i-1)*M+j-1] + t[(i-1)*M+j] + t[(i-1)*M+j+1] + t[i*M+j+1] + t[0*M+j+1] + t[0*M+j] + t[0*M+j-1] + t[i*M+j-1];
			}
        }
        else if (i-1 < 0) {
            if (j-1 < 0) {
                /* eimai o top_left */
				neighbours = t[(N-1)*M+M-1] + t[(N-1)*M+j] + t[(N-1)*M+j+1] + t[i*M+j+1] + t[(i+1)*M+j+1] + t[(i+1)*M+j] + t[(i+1)*M+M-1] + t[i*M+M-1];
			}
            else if (j+1 > M-1) {
                /* eimai o top_right */
				neighbours = t[(N-1)*M+j-1] + t[(N-1)*M+j] + t[(N-1)*M+0] + t[i*M+0] + t[(i+1)*M+0] + t[(i+1)*M+j] + t[(i+1)*M+j-1] + t[i*M+j-1];
			}
            else{
                /* eimai aplos top */
				neighbours = t[(N-1)*M+j-1] + t[(N-1)*M+j] + t[(N-1)*M+j+1] + t[i*M+j+1] + t[(i+1)*M+j+1] + t[(i+1)*M+j] + t[(i+1)*M+j-1] + t[i*M+j-1];
			}
        }
        else if (j-1 < 0) {
            /* eimai aplos left */
			neighbours = t[(i-1)*M+M-1] + t[(i-1)*M+j] + t[(i-1)*M+j+1] + t[i*M+j+1] + t[(i+1)*M+j+1] + t[(i+1)*M+j] + t[(i+1)*M+M-1] + t[i*M+M-1];
		}
        else if (j+1 > M-1) {
            /* eimai aplos right */
			neighbours = t[(i-1)*M+j-1] + t[(i-1)*M+j] + t[(i-1)*M+0] + t[i*M+0] + t[(i+1)*M+0] + t[(i+1)*M+j] + t[(i+1)*M+j-1] + t[i*M+j-1];
		}
        else{
            /* oi geitones mou den peftoun eksw */
			neighbours = t[(i-1)*M+j-1] + t[(i-1)*M+j] + t[(i-1)*M+j+1] + t[i*M+j+1] + t[(i+1)*M+j+1] + t[(i+1)*M+j] + t[(i+1)*M+j-1] + t[i*M+j-1];
		}
        /* kanones paixnidiou edw */
        if (t[x] == ALIVE) {
            if (neighbours <= 1 || neighbours >= 4) {
                t1[x] = DEAD;
            }
            else{
                t1[x] = ALIVE;
            }
        }
        else if (t[x] == DEAD && neighbours == 3) {
            t1[x] = ALIVE;
        }
		else{
			t1[x] = DEAD;
		}
    }
}

__global__ void initdat(int *t, int *t1, int N, int M, time_t clock){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	curandState_t state;
	curand_init(clock,x,0,&state);


	if (0 <= x && x < N*M ){
		t[x] = (curand(&state) % 4) ? DEAD : ALIVE;
		t1[x] = DEAD;
	}

}
