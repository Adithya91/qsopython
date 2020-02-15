__kernel void convolve
	(
		__global double * input,    // Matrix input
		__constant uint * rows_sizes, 	// Table of the number of significant elements
					   	// from the next lines of the input matrix
		__constant double * filter, // Filter, table 1d
		uint filter_size,          // Filter size
		__global double * output    // Result matrix
	)
{
	// gid0 - Quasar spectrum number
	uint gid0 = get_global_id(0);
	// gid0 - index in the spectrum
	uint gid1 = get_global_id(1);	
	uint gid1_size = get_global_size(1);
	
	uint this_row_size;
	__local uint this_row_size_local;
	if (get_local_id(1) == 0)
	{
  		this_row_size_local = rows_sizes[gid0];
	}
	barrier(CLK_LOCAL_MEM_FENCE);	
	this_row_size = this_row_size_local;

	/// The filter is above the item
	// which is no longer significant (no need to calculate).
	if(gid1 >= this_row_size)
	{
		return;
	}

	// Correction due to filter length (even / odd).
	int fix = select(1, 0, filter_size%2 == 0);
	// Center
	int filter_center_idx  = ((filter_size+fix) / 2) - 1;
	
	// Calculate the filter index for which we start.
	// We start with further filter values ​​and go back to the zero index.
	// But the signal index will increase.
	int filter_idx = gid1 - filter_center_idx;
	filter_idx = select(((int)filter_size)-1, ((int)filter_size)-1+filter_idx, filter_idx < 0);
	
	// Calculation of the index for the signal.
	int idx = gid1 - filter_center_idx - abs(1-fix);
	idx = select(idx, 0, idx < 0);		
	// And we skip the previous signals.
	idx += gid0 * gid1_size;
	int idx_max = gid0 * gid1_size + this_row_size;
	
	//if(gid0 == 0 && (gid1 < 10 || gid1 > 4090))
	//{
	//	printf("gid1 = %d\n", gid1);
	//	printf("idx = %d\n", idx);
	//	printf("idx_max = %d\n", idx_max);
	//	printf("filter_idx = %d\n", filter_idx);
	//}	
	
	double sum = 0;
	while(filter_idx >= 0 && idx < idx_max)
	{
		//if(gid1 < 64 && idx < 64)
		//{
		//	printf("gid1 = %d, idx = %d\n", gid1, idx);
		//}	
		
		sum += (input[idx] * filter[filter_idx]);
		filter_idx--;
		idx++;
	}
	
	output[gid0 * gid1_size + gid1] = sum;
	
	return;
}

// Copies the value of the corresponding value to the next lines of the matrix
// line with input values ​​that are different from +/- INFINITY.
// Does not create "gaps" in the rows of the output matrix.
//
__kernel void copyIfNotInf
	(
		__global double * input,	// Input matrix.
		uint width,
		uint height,
		__global double * output,// Result matrix.
		uint output_height	// The size of the output matrix row.
	)
{
	uint gid = get_global_id(0);		
	if(gid >= width)
	{
		return;
	}

	// index 
	uint input_idx = gid;
	uint input_end = input_idx + height * width;

	uint output_idx = gid;
	uint output_end = output_idx + output_height * width;

	for(; input_idx < input_end && output_idx < output_end; input_idx+=width)
	{
		double in = input[input_idx];
		if(in != INFINITY && in != -INFINITY)
		{			
			output[output_idx] = in;
			output_idx+=width;
		}
		
	}	
}

// Count the occurrences of items other than +/- INFINITY
// on the lines of the input matrix.
// Writes the results to the result array.
__kernel void countIfNotInf
	(
		__global double * input,	// Input matrix.
		uint width,
		uint height,
    	__global uint * result  // Results, number of elements! = INF for each line.
	) 
{
	uint gid = get_global_id(0);		
	if(gid >= width)
	{
		return;
	}

	uint idx = gid;
	uint idx_max = idx + height * width;
	
	uint count = 0;
	uint add = 0;
	double in;
	while(idx < idx_max)
	{
		in = input[idx];
		add = select(0, 1, in != INFINITY); 
		count += add;
		idx += width;
	}
	result[gid] = count;
}

// 
#if defined(cl_khr_fp64)
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
	// double
	typedef double real_t;
	typedef double2 real2_t;
	typedef double4 real4_t;
#else
	// double
	typedef double real_t;
	typedef double2 real2_t;
	typedef double4 real4_t;
#endif

//
// Calculates simple regression coefficients.
//
//
// Result structure: (a, b, sia2, sib2, siy2, x_sum, x ^ 2_sum, y_sum)
__kernel void reglin
	(
    	__global double * xs,	// Matrix xs
		__global double * ys,	// Matrix ys
		uint width,
		uint height,
		__constant uint * cols_sizes, 	// Table of the number of significant elements
					   	// from the next columns of the input matrix
		__global double8 * results	// An array with all coefficients for each row.
	) 
{
	uint gid = get_global_id(0);		
	if(gid >= width)
	{
		return;
	}

	uint idx = gid;
	uint col_height = cols_sizes[gid];
	uint idx_max = idx + col_height * width;

	real_t x_sum = 0;
	real_t x2_sum = 0;
	real_t y_sum = 0;
	real_t xy_sum = 0;
	
#if defined(cl_khr_fp64)
	real_t n = convert_double(col_height);
#else
	real_t n = convert_double(col_height);
#endif

	{
		real4_t c = (real4_t)(0.0f, 0.0f, 0.0f, 0.0f);
		while (idx < idx_max) 
		{
			real4_t t, u;
			real_t x, y;

			x = xs[idx];
			y = ys[idx];

			u = (real4_t)(x, pown(x, 2), y, x*y) - c;			
			t = (real4_t)(x_sum, x2_sum, y_sum, xy_sum) + u;
			c = (t - (real4_t)(x_sum, x2_sum, y_sum, xy_sum)) - u;

			x_sum 	= t.s0;
			x2_sum 	= t.s1;
			y_sum	= t.s2;
			xy_sum	= t.s3;

			idx += width;
		}
	}
	// s0 - x_sum
	// s1 - x2_sum
	// s2 - y_sum
	// s3 - xy_sum

	// Calculation of coefficients a and b
	real_t dd = ( n * x2_sum) - pown(x_sum, 2);
	double b = convert_double( ( (x2_sum * y_sum) - (x_sum * xy_sum) ) / dd );
	double a = convert_double( ( (n * xy_sum) - (x_sum * y_sum) ) / dd );

	// CALCULATION OF THE SUM NECESSARY TO CALCULATE SIY2
	idx = gid;
	real_t siy2 = 0;

	real_t c = 0;
	while (idx < idx_max) 
	{
		real_t x, y, t, u;
		x = xs[idx];
		y = ys[idx];
		
		u = pown((y - b - (a * x)), 2) - c;
		t = siy2 + u;
		c = (t - siy2) - u;
		
		siy2 = t;

		idx += width;
	}

	// SAVING RESULTS TO THE GLOBAL MEMORY		
	siy2 = ( (real_t)(1.0)/(n - (real_t)(2.0))) * siy2;		
	results[gid] = (double8)(a, b, 
				 pow((n * siy2 / dd), (real_t)(0.5)),
				 pow((siy2 * x2_sum / dd), (real_t)(0.5)),
				 pow(siy2, (real_t)(0.5)),
				 x_sum, x2_sum, y_sum);
}

//
// Calculates the coefficient a of the simple regression y = a * x.
//
//
// Result structure: (a)
__kernel void reglin_yax
	(
    	__global double * xs,	// Matrix xs
		__global double * ys,	// Matrix ys
		uint width,
		uint height,
		__constant uint * cols_sizes, 	// Table of the number of significant elements
					   	// from the next columns of the input matrix
		__global double * results	// An array with all coefficients for each row.
	) 
{
	uint gid = get_global_id(0);		
	if(gid >= width)
	{
		return;
	}

	uint idx = gid;
	uint col_height = cols_sizes[gid];
	uint idx_max = idx + col_height * width;

	real_t x2_sum = 0;
	real_t xy_sum = 0;	
	{
		real2_t c = (real2_t)(0, 0);
		while (idx < idx_max) 
		{
			real2_t t, u;
			real_t x, y;

			x = xs[idx];
			y = ys[idx];

			u = (real2_t)(pown(x, 2), x*y) - c;			
			t = (real2_t)(x2_sum, xy_sum) + u;
			c = (t - (real2_t)(x2_sum,  xy_sum)) - u;

			x2_sum 	= t.s0;
			xy_sum 	= t.s1;

			idx += width;
		}
	}
	real_t r = (xy_sum / x2_sum);
	if(isnan(r))
	{
		r = (real_t)(0);
	}	
	results[gid] = (double)(r);
}

#if defined(cl_khr_fp64)
	#pragma OPENCL EXTENSION cl_khr_fp64 : disable
#endif

// Calculate chisq between corresponding columns
// from the matrix fs and ys.
//
__kernel void chisq
	(
    	__global double * fs,	// Matrix of fs
		__global double * ys,	// Matrix ys
		__global double * errs,	// Matrix error  
		uint width,
		uint height,
		__constant uint * cols_sizes, 	// Table of the number of significant elements
					   	// from the next columns of the input matrix
		__global double * results	// Table with chisq results /
	) 
{
	uint gid = get_global_id(0);		
	if(gid >= width)
	{
		return;
	}

	uint idx = gid;
	uint col_height = cols_sizes[gid];
	uint idx_max = idx + col_height * width;

	double chi2 = 0.0f;
	double c = 0.0f;

	while (idx < idx_max) 
	{
		double f, y, e, t, u;

		f = fs[idx];
		y = ys[idx];
		e = errs[idx];

		u = pown((y-f),2) / pown(e, 2) - c;
		t = chi2 + u;
		c = (t - chi2) - u;

		chi2 = t;

		idx += width;
	}		
	results[gid] = chi2;
}


// Calculates the approximation of the integral using the trapezodial rule for many
// functions simultaneously.
//
// The values ​​of the integral functions f (x) are in the columns of the matrix ys,
// and the arguments of these functions in the corresponding columns of the xs matrix.
//
__kernel void integrate_trapz
	(
    	__global double * ys,	// The y matrix, each column is a set of values ​​of the function f (x).
		__global double * xs,	//Matrix x, each column is a set of arguments to the function.
		uint width,		//Matrix width.
		uint height,		// Matrix height.
		__constant uint * cols_sizes, 	// An array with the number of f (x) values ​​for each column
					   	// matrix ys.
		__global double * integrals	// Table with integration results.
	) 
{
	uint gid = get_global_id(0);		
	if(gid >= width)
	{
		return;
	}

	uint idx = gid;
	uint col_height = cols_sizes[gid];
	uint idx_max = idx + col_height * width;

	double integral = 0.0f;
	double c = 0.0f;

	double fx1, fx2;
	double x1, x2;

	fx1 = ys[idx];
	x1 = xs[idx];
	idx += width;

	while (idx < idx_max) 
	{
		double t, u;

		fx2 = ys[idx];
		x2 = xs[idx];

		u = (((fx1 + fx2) / 2.0f) * fabs(x2 - x1)) - c;
		t = integral + u;
		c = (t - integral) - u;

		integral = t;

		fx1 = fx2;
		x1 = x2;

		idx += width;
	}		
	integrals[gid] = integral;
}








