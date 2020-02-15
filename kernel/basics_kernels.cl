
__kernel void matrix_log10
	(
		__global double * matrix,	
		uint row_size			// Matrix row size.
	)
{
	// gid0 - line number of input matrix
	uint gid0 = get_global_id(0);
	// gid1 - item number in the line.
	uint gid1 = get_global_id(1);

	if(gid1 >= row_size)
	{
		return;
	}
	
	uint idx = gid0 * row_size + gid1;

	double m = matrix[idx];
	m = log10(m);
	matrix[idx] = m;
}

__kernel void matrix_minus_scalar
	(
		__global double * matrix,	// Matrix
		uint row_size,			// Matrix row size.
		double subtrahend		// Number to subtract
	)
{
	// gid0 - line number of input matrix
	uint gid0 = get_global_id(0);
	// gid1 - item number in the line.
	uint gid1 = get_global_id(1);

	if(gid1 >= row_size)
	{
		return;
	}
	
	uint idx = gid0 * row_size + gid1;

	double m = matrix[idx];
	m -= subtrahend;
	matrix[idx] = m;
}


__kernel void matrix_plus_matrix
	(
		__global double * matrix,	// Matrix
		uint row_size,			// Matrix row size.
		__global double * matrix_to_add,		// Matrix to subtract
		__global double * output_matrix	// Output
	)
{
	// gid0 - line number of input matrix
	uint gid0 = get_global_id(0);
	// gid1 - item number in the line.
	uint gid1 = get_global_id(1);

	if(gid1 >= row_size)
	{
		return;
	}
	
	uint idx = gid0 * row_size + gid1;

	double value = matrix[idx];
	double value_to_add = matrix_to_add[idx];

	value += value_to_add;

	output_matrix[idx] = value;
}

__kernel void matrix_minus_matrix
	(
		__global double * matrix,	
		uint row_size,			
		__global double * subtrahend_matrix,		// Matrix to subtract
		__global double * output_matrix	
	)
{
	
	uint gid0 = get_global_id(0);
	
	uint gid1 = get_global_id(1);

	if(gid1 >= row_size)
	{
		return;
	}
	
	uint idx = gid0 * row_size + gid1;

	double value = matrix[idx];
	double subtrahend = subtrahend_matrix[idx];

	value -= subtrahend;

	output_matrix[idx] = value;
}

__kernel void matrix_divide_matrix
	(
		__global double * dividend_matrix,	
		uint row_size,				
		__global double * divisor_matrix,	
		__global double * output_matrix		
	)
{
	
	uint gid0 = get_global_id(0);
	
	uint gid1 = get_global_id(1);

	if(gid1 >= row_size)
	{
		return;
	}
	
	uint idx = gid0 * row_size + gid1;
        
	double dividend = dividend_matrix[idx];
	double divisor = divisor_matrix[idx];

	output_matrix[idx] = dividend /= divisor;
}

// Multiplies every element of the i-th column by the i-th element
// given vector.
//
__kernel void matrix_multiply_col_vector
	(
		__global double * matrix,	
		uint row_size,			
		__constant double * vector,	// A vector that contains at least
						// as many elements as the columns have matrix.
		__global double * output_matrix	
	)
{
	
	uint gid0 = get_global_id(0);
	
	uint gid1 = get_global_id(1);

	if(gid1 >= row_size)
	{
		return;
	}
	
	uint idx = gid0 * row_size + gid1;
	uint col_idx = gid1;

	double value = matrix[idx];
	double m = vector[col_idx];

	value *= m;

	output_matrix[idx] = value;
}

__kernel void matrix_transpose1
	(
		__global double * matrix,	// Input matrix
		__global double * tmatrix,	// Transposed matrix
		uint width,			// Number of columns, matrix width
		uint height,			// Number of rows, height of the matrix
		__local double * scratch
	)
{
	
	uint x_idx = get_global_id(0);
	
	uint y_idx = get_global_id(1);
	uint M_TRANSPOSE_BLOCK_DIM = 16;
	
	uint idx;

		
	if((x_idx < width) && (y_idx < height))
	{	
		idx = y_idx * width + x_idx;
		scratch[get_local_id(1)*(M_TRANSPOSE_BLOCK_DIM+1)+get_local_id(0)] = matrix[idx];
	}
	barrier(CLK_LOCAL_MEM_FENCE);


	// Retrieving values ​​from matrix	
	x_idx = get_group_id(1) * M_TRANSPOSE_BLOCK_DIM + get_local_id(0);
	y_idx = get_group_id(0) * M_TRANSPOSE_BLOCK_DIM + get_local_id(1);
	if((x_idx < height) && (y_idx < width))
	{	
		idx = y_idx * height + x_idx;
		tmatrix[idx] = scratch[get_local_id(0)*(M_TRANSPOSE_BLOCK_DIM+1)+get_local_id(1)];
	}
}



#define BLOCK_SIZE 16
#define A_BLOCK_STRIDE (BLOCK_SIZE * a_height)
#define A_T_BLOCK_STRIDE (BLOCK_SIZE * a_width)

__kernel void matrix_transpose2(
		__global double *a,
        __global double *a_t, 
        unsigned a_width, unsigned a_height,
        __local double *a_local)
{
        int base_idx_a   =
            get_group_id(0) * BLOCK_SIZE +
            get_group_id(1) * A_BLOCK_STRIDE;
        int base_idx_a_t =
            get_group_id(1) * BLOCK_SIZE +
            get_group_id(0) * A_T_BLOCK_STRIDE;

        int glob_idx_a   = base_idx_a + get_local_id(0) + a_width * get_local_id(1);
        int glob_idx_a_t = base_idx_a_t + get_local_id(0) + a_height * get_local_id(1);

        a_local[get_local_id(1)*BLOCK_SIZE+get_local_id(0)] = a[glob_idx_a];

        barrier(CLK_LOCAL_MEM_FENCE);

        a_t[glob_idx_a_t] = a_local[get_local_id(0)*BLOCK_SIZE+get_local_id(1)];
}


// printf("h w %d %d indeksy %d %d wektory %d %d matrix %f ", a_height, a_width, get_global_id(0), get_global_id(1), read_idx, write_idx, a[read_idx]);

__kernel void matrix_transpose3(
		  __global double *a,
          __global double *a_t, 
		  unsigned a_height,
          unsigned a_width 
		  )
{
          int read_idx = get_global_id(1) + get_global_id(0) * a_width;
          int write_idx = get_global_id(0) + get_global_id(1) * a_height;
          a_t[write_idx] = a[read_idx];
}






















