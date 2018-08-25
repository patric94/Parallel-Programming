#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>

#define ALIVE 1
#define DEAD 0

#define TAG_EVAL_TOP 32
#define TAG_EVAL_BOT 33
#define TAG_EVAL_LEFT 34
#define TAG_EVAL_RIGHT 35
#define TAG_EVAL_TOPLEFT 36
#define TAG_EVAL_TOPRIGHT 37
#define TAG_EVAL_BOTLEFT 38
#define TAG_EVAL_BOTRIGHT 39

typedef struct Block {
	int rows;
	int cols;
	int* grid;
}Block;


int MyMPI_Cart_shift(MPI_Comm comm, int* top_right, int* top_left, int* bot_right, int* bot_left);
int divide_into_blocks(int N, int M, int procs, int* rows, int* cols);
void print_block(Block *block);
int* offset(int *grid, int startrow, int startcol, int length);
void update(int *prev, int *next, int starting_row, int ending_row, int starting_col, int ending_col, int width);
int diffGrids(Block *prev, Block *next);
int is_prime(int num);
void MyMPI_WaitanyAndUpdate(int count, MPI_Request** array_of_requests, MPI_Status* status, Block* block, Block* next_block);

int main(int argc, char *argv[]) {

	int size,			/* number of processes */
		my_rank,		/* process' rank */
		N,				/* rows of the grid */
	 	M,				/* columns of the grid */
		Gens,			/* amount of generations */
		perGens,		/* checks termination per perGens generations. if perGens zero no termination check */
		rc;				/* return code */
	Block *block, *next_block, *grid=NULL;
	int* temp;


	block = (Block*)malloc(sizeof(Block));
	next_block = (Block*)malloc(sizeof(Block));

	srand(time(NULL)*getpid());

	if ( argc != 5) {
		printf("Error! Missing mandatory argument.\n");
		return 1;
	}

	N = atoi(argv[1]); /* Getting rows amount */
	M = atoi(argv[2]); /* Getting columns amount */
	Gens = atoi(argv[3]); /* Getting Gens */
	perGens = atoi(argv[4]); /* Getting perGens */
	if (Gens <= 0 || N < 0 || M < 0 || perGens < 0) {
		printf("Please give positive values for rows/cols and Generations and perGens\n");
		return 1;
	}

	/* Start up MPI */
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

	if (my_rank == 0) {
		if (size != 2 && is_prime(size)) {
			printf("Doesnt work for a prime number of processes.\n");
			MPI_Abort(MPI_COMM_WORLD,rc);
			exit(1);
		}
	}

	/* MPI data types */
	MPI_Datatype col_type;
	MPI_Datatype row_type;
	// MPI_Datatype subarray;

	/* MPI requests */
	MPI_Request send_top_req;
	MPI_Request send_bot_req;
	MPI_Request send_left_req;
	MPI_Request send_right_req;
	MPI_Request send_topleft_req;
	MPI_Request send_topright_req;
	MPI_Request send_botleft_req;
	MPI_Request send_botright_req;
	MPI_Request recv_top_req;
	MPI_Request recv_bot_req;
	MPI_Request recv_left_req;
	MPI_Request recv_right_req;
	MPI_Request recv_topleft_req;
	MPI_Request recv_topright_req;
	MPI_Request recv_botleft_req;
	MPI_Request recv_botright_req;
	// MPI_Request* send_reqs[8] = {send_top_req,send_bot_req,send_left_req,send_right_req,send_topleft_req,send_topright_req,send_botleft_req,send_botright_req};
	MPI_Request* recv_reqs[8] = {&recv_top_req,&recv_bot_req,&recv_left_req,&recv_right_req,&recv_topleft_req,&recv_topright_req,&recv_botleft_req,&recv_botright_req};

	//grid = (Block*)malloc(sizeof(Block)); // this makes sense only for process 0, but needs to be seen from every process for MPI_Gather to work with the struct Block.
	if (my_rank == 0) {
		/************************* master code *******************************/
		if (divide_into_blocks(N, M, size, &(block->rows), &(block->cols)) < 0) {
			printf("Cannot divide into blocks\n");
	    	MPI_Abort(MPI_COMM_WORLD, rc);
			exit(1);
		}

		next_block->rows = block->rows;
		next_block->cols = block->cols;

		//grid->cols = M;
		//grid->rows = N;
		//grid->grid = (int*)malloc(sizeof(int)*(grid->cols)*(grid->rows));
	}
	/* Broadcast parameters */
	MPI_Bcast(&(block->rows), 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&(block->cols), 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&(next_block->rows), 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&(next_block->cols), 1, MPI_INT, 0, MPI_COMM_WORLD);

	/* Create column data type */
	MPI_Type_vector(block->rows-2, 1, block->cols, MPI_INT, &col_type);
	MPI_Type_commit(&col_type);
	/* Create row data type */
	MPI_Type_contiguous(block->cols-2, MPI_INT, &row_type);
	MPI_Type_commit(&row_type);
	/* create subarray data type */
	// int startpos[2] = {1, 1};
	// int subsize[2] = {block->rows-2, block->cols-2};
	// int startsize[2] = {block->rows, block->cols};
	// MPI_Type_create_subarray(2, startsize, subsize, startpos, MPI_ORDER_C, MPI_INT, &subarray);
	// MPI_Type_commit(&subarray);

	MPI_Comm cart_comm;
    int dim[2], period[2], reorder;
	/* Initialising processes' neighbours */
	int top = -1,
	 	bot = -1,
		right = -1,
		left = -1,
		top_left = -1,
		top_right = -1,
		bot_left = -1,
		bot_right = -1;

    dim[0] = N/(block->rows-2);
    dim[1] = M/(block->cols-2);
    period[0] = 1;
    period[1] = 1;
    reorder = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &cart_comm);

    MPI_Cart_shift(cart_comm, 0, 1, &top, &bot);
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);
	MyMPI_Cart_shift(cart_comm, &top_right, &top_left, &bot_right, &bot_left);

	/* Init block */
	block->grid = (int*)calloc((block->rows)*(block->cols), sizeof(int));
	next_block->grid = (int*)calloc((block->rows)*(block->cols), sizeof(int));

	for (int i = 1; i <= block->rows-2; i++) {
		for (int j = 1; j <= block->cols-2; j++) {
			block->grid[i*(block->cols)+j] = (rand() % 4) ? DEAD : ALIVE;
		}
	}

	// MPI_Gather(block->grid, 1, subarray, grid->grid, (block->cols-2)*(block->rows-2), MPI_INT, 0, cart_comm);

	MPI_Barrier(cart_comm);
	int gen = 0;

	double finish_time, start_time;
	start_time = MPI_Wtime();

	while (1) {
		gen++;
		/* compute grid offsets for neighbours */
		int *sendTop = offset(block->grid, 1, 1, block->cols);
		int *sendBot = offset(block->grid, block->rows-2, 1, block->cols);
		int *sendLeft = offset(block->grid, 1, 1, block->cols);
		int *sendRight = offset(block->grid, 1, block->cols-2, block->cols);

		int *recvTop = offset(block->grid, 0, 1, block->cols);
		int *recvBot = offset(block->grid, block->rows-1, 1, block->cols);
		int *recvLeft = offset(block->grid, 1, 0, block->cols);
		int *recvRight = offset(block->grid, 1, block->cols-1, block->cols);

		int *sendTopLeft = offset(block->grid, 1, 1, block->cols);
		int *sendTopRight = offset(block->grid, 1, block->cols-2, block->cols);
		int *sendBotLeft = offset(block->grid, block->rows-2, 1, block->cols);
		int *sendBotRight = offset(block->grid, block->rows-2, block->cols-2, block->cols);

		int *recvTopLeft = offset(block->grid, 0, 0, block->cols);
		int *recvTopRight = offset(block->grid, 0, block->cols-1, block->cols);
		int *recvBotLeft = offset(block->grid, block->rows-1, 0, block->cols);
		int *recvBotRight = offset(block->grid, block->rows-1, block->cols-1, block->cols);

		/* Send and Receive rows/cols from neighbour processes */
		/* To/From North */
		MPI_Isend(sendTop, 1, row_type, top, TAG_EVAL_TOP, cart_comm, &send_top_req);
		MPI_Irecv(recvTop, 1, row_type, top, TAG_EVAL_BOT, cart_comm, &recv_top_req);
		/* To/From South */
		MPI_Isend(sendBot, 1, row_type, bot, TAG_EVAL_BOT, cart_comm, &send_bot_req);
		MPI_Irecv(recvBot, 1, row_type, bot, TAG_EVAL_TOP, cart_comm, &recv_bot_req);
		/* To/From West */
		MPI_Isend(sendLeft, 1, col_type, left, TAG_EVAL_LEFT, cart_comm, &send_left_req);
		MPI_Irecv(recvLeft, 1, col_type, left, TAG_EVAL_RIGHT, cart_comm, &recv_left_req);
		/* To/From East */
		MPI_Isend(sendRight, 1, col_type,  right, TAG_EVAL_RIGHT, cart_comm, &send_right_req);
		MPI_Irecv(recvRight, 1, col_type,  right, TAG_EVAL_LEFT, cart_comm, &recv_right_req);
		/* To/From NorthWest */
		MPI_Isend(sendTopLeft, 1, MPI_INT, top_left, TAG_EVAL_TOPLEFT, cart_comm, &send_topleft_req);
		MPI_Irecv(recvTopLeft, 1, MPI_INT, top_left, TAG_EVAL_BOTRIGHT, cart_comm, &recv_topleft_req);
		/* To/From SouthWest */
		MPI_Isend(sendBotLeft, 1, MPI_INT, bot_left, TAG_EVAL_BOTLEFT, cart_comm, &send_botleft_req);
		MPI_Irecv(recvBotLeft, 1, MPI_INT, bot_left, TAG_EVAL_TOPRIGHT, cart_comm, &recv_botleft_req);
		/* To/From NorthEast */
		MPI_Isend(sendTopRight, 1, MPI_INT, top_right, TAG_EVAL_TOPRIGHT, cart_comm, &send_topright_req);
		MPI_Irecv(recvTopRight, 1, MPI_INT, top_right, TAG_EVAL_BOTLEFT, cart_comm, &recv_topright_req);
		/* To/From SouthEast */
		MPI_Isend(sendBotRight, 1, MPI_INT, bot_right, TAG_EVAL_BOTRIGHT, cart_comm, &send_botright_req);
		MPI_Irecv(recvBotRight, 1, MPI_INT, bot_right, TAG_EVAL_TOPLEFT, cart_comm, &recv_botright_req);


		/* Inner computations */
		update(block->grid, next_block->grid, 2, block->rows-3, 2, block->cols-3, block->cols);

		for (int i = 0; i < 8; i++) {
			MyMPI_WaitanyAndUpdate(8,recv_reqs,MPI_STATUS_IGNORE,block,next_block);
		}

		/* Check if the game must end */
		if (gen == Gens) {
			if (my_rank == 0) {
				printf("Reached requested Generations %d!\n", Gens);
			}
			break;
		}
		else if ( perGens) {
			if (gen % perGens == 0) {
				int isDiff = diffGrids(block, next_block);
				int allDiff;
				MPI_Allreduce(&isDiff, &allDiff, 1, MPI_INT, MPI_SUM, cart_comm);
				if (!allDiff) {
					/* The grid is the same after one generation */
					if (my_rank == 0) {
						printf("Generation %d is the same with generation %d\n", gen, gen-1);
					}
					break;
				}
				/* Calculate the sum of each block of the grid */
				int local_sum = 0;
				for (int i = 1; i < next_block->rows-2; i++) {
					for (int j = 1; j < next_block->cols-2; j++) {
						local_sum += next_block->grid[i*next_block->cols + j];
					}
				}
				int global_sum;
				/* Calculate the total sum of the grid and redistribute it back to all processes */
				MPI_Allreduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, cart_comm);

				if (global_sum == 0) {
					/* There is no alive organism in the grid */
					if (my_rank == 0) {
						printf("All organisms are dead!\n");
					}
					break;
				}
			}
		}

		/* Swap blocks */
		temp = block->grid;
		block->grid = next_block->grid;
		next_block->grid = temp;
	}

	finish_time = MPI_Wtime();

	double total_time = finish_time - start_time;
	double *times;
	if(my_rank == 0){
		times = malloc(sizeof(double)*size);
	}

	MPI_Gather(&total_time, 1, MPI_DOUBLE, times, 1, MPI_DOUBLE, 0, cart_comm);

	if(my_rank == 0){
		float max = times[0];
		for(int i = 1; i < size; i++)
			if(times[i] < max)
				max = times[i];
		printf("CPU TIME: %.5f secs\n", max);
		free(times);
	}

	free(block->grid);
	free(next_block->grid);
	free(block);
	free(next_block);
	// free(grid);
	// if (my_rank == 0) {
	// 	free(grid->grid);
	// }

	/* Shut down MPI */
    MPI_Type_free(&col_type);
    MPI_Type_free(&row_type);
    // MPI_Type_free(&subarray);
	MPI_Finalize();
	return 0;
}

/* Function to find the corner neighbours of a process */
int MyMPI_Cart_shift(MPI_Comm comm, int* top_right, int* top_left, int* bot_right, int* bot_left){
	int my_rank, my_coord1, my_coord2, ndims, *coords;

	MPI_Cartdim_get(comm, &ndims);
	coords = calloc(ndims, sizeof(int));

	MPI_Comm_rank(comm, &my_rank);
	MPI_Cart_coords(comm, my_rank, ndims, coords);
	my_coord1 = coords[0];
	my_coord2 = coords[1];
	coords[0] = my_coord1 - 1;
	coords[1] = my_coord2 - 1;
	MPI_Cart_rank(comm, coords, top_left);
	coords[0] = my_coord1 - 1;
	coords[1] = my_coord2 + 1;
	MPI_Cart_rank(comm, coords, top_right);
	coords[0] = my_coord1 + 1;
	coords[1] = my_coord2 - 1;
	MPI_Cart_rank(comm, coords, bot_left);
	coords[0] = my_coord1 + 1;
	coords[1] = my_coord2 + 1;
	MPI_Cart_rank(comm, coords, bot_right);

	free(coords);
	return MPI_SUCCESS;
}

void MyMPI_WaitanyAndUpdate(int count, MPI_Request** array_of_requests, MPI_Status* status, Block* block, Block* next_block){
	int index;
	MPI_Waitany(count,*array_of_requests,&index,status);
	if(index == 0){
		update(block->grid, next_block->grid, 1, 1, 2, block->cols-3, block->cols);
	}
	else if (index == 1) {
		update(block->grid, next_block->grid, block->rows-2, block->rows-2, 2, block->cols-3, block->cols);
	}
	else if (index == 2) {
		update(block->grid, next_block->grid, 2, block->rows-3, 1, 1, block->cols);
	}
	else if (index == 3) {
		update(block->grid, next_block->grid, 2, block->rows-3, block->cols-2, block->cols-2, block->cols);
	}
	else if (index == 4) {
		update(block->grid, next_block->grid, 1, 1, 1, 1, block->cols);
	}
	else if (index == 5) {
		update(block->grid, next_block->grid, 1, 1, block->cols-2, block->cols-2, block->cols);
	}
	else if (index == 6) {
		update(block->grid, next_block->grid, block->rows-2, block->rows-2, 1, 1, block->cols);
	}
	else if (index == 7) {
		update(block->grid, next_block->grid, block->rows-2, block->rows-2, block->cols-2, block->cols-2, block->cols);
	}
}

int divide_into_blocks(int N, int M, int procs, int* rows, int* cols){
	int total_cells, block_cells, i;

	total_cells = N*M;
	if (total_cells % procs != 0) {
		return -1;
	}
	block_cells = total_cells / procs;		// total_cells will be a multiple of the number of processes. *CONVENTION*
	*rows = 1;
	*cols = block_cells;

	for ( i = 1; i <= (int)sqrt(block_cells) ; i++) {
		if (block_cells % i == 0) {
			if ((M % i == 0) && (N % (block_cells/i) == 0)) {
				*rows = block_cells/i;
				*cols = i;
			}
			else if ((N % i == 0) && (M % (block_cells/i) == 0)) {
				*cols = block_cells/i;
				*rows = i;
			}
		}
	}
	*cols += 2;
	*rows += 2;
	return 1;
}

void print_block(Block *block){
	/* Ektyposi periexomenou epauximenou block */
	for (int i = 0; i < (block->rows)*(block->cols); i++) {
		if (i % (block->cols) == 0) {
			printf("\n");
		}
		printf("%d ", block->grid[i] );
	}
	printf("\n");

}

int* offset(int *grid, int startrow, int startcol, int length) {
    return &grid[length * startrow + startcol];
}

void update(int *prev, int *next, int starting_row, int ending_row, int starting_col, int ending_col, int width){
	int i,j,neighbours;

	for (i = starting_row; i <= ending_row ; i++) {
		for (j = starting_col; j <= ending_col; j++) {
			neighbours = prev[(i-1)*width + j-1] + prev[(i-1)*width + j] + prev[(i-1)*width + j+1] + prev[i*width + j-1] + prev[i*width + j+1] + prev[(i+1)*width + j-1] + prev[(i+1)*width + j] + prev[(i+1)*width + j+1];
			if (prev[i*width + j] == ALIVE) {
				if (neighbours <= 1 || neighbours >= 4) {
					next[i*width + j] = DEAD;
				}
				else{
					next[i*width + j] = ALIVE;
				}
			}
			else if (prev[i*width + j] == DEAD && neighbours == 3) {
				next[i*width + j] = ALIVE;
			}
			else {
				next[i*width + j] = DEAD;
			}
		}
	}
}

int diffGrids(Block *prev, Block *next){
	for (int i = 1; i < prev->rows-2; i++) {
		for (int j = 1; j < prev->cols-2; j++) {
			if (prev->grid[i*prev->cols+j] != next->grid[i*prev->cols+j]) {
				return 1;
			}
		}
	}
	return 0;
}

int is_prime(int num){
     if (num <= 1) return 0;
     if (num % 2 == 0 && num > 2) return 0;
     for(int i = 3; i <= floor(sqrt(num)); i+= 2)
     {
         if (num % i == 0)
             return 0;
     }
     return 1;
}
