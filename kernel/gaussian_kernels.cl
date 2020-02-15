
// 
//#if defined(cl_khr_fp64)
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
	// double
	typedef double real_t;
	typedef double2 real2_t;
	typedef double4 real4_t;
	#define REAL_MIN DBL_MIN
//#else
	// double
//	typedef float real_t;
//	typedef float2 real2_t;
//	typedef float4 real4_t;
//	#define REAL_MIN FLT_MIN
//#endif

real_t max_in_vector(real4_t vector)
{
	return max(max(vector.x, vector.y), max(vector.z, vector.w));
}

// Gaussian
real_t f(real_t x, real_t a, real_t b, real_t c)
{
	return a * exp( -0.5 * pown((x - b), 2) / pown(c, 2));
}

// dfda
real_t dfda(real_t x, real_t b, real_t c)
{
	return exp( -0.5 * pown((x - b), 2) / pown(c, 2));
}

// dfdb
real_t dfdb(real_t x, real_t a, real_t b, real_t c)
{
	return ((a * (x - b)) / pown(c, 2)) * exp( -0.5 * pown((x - b), 2) / pown(c, 2));
}

// dfdc
real_t dfdc(real_t x, real_t a, real_t b, real_t c)
{
	return ((a * pown((x - b), 2)) / pown(c, 3)) * exp( -0.5 * pown((x - b), 2) / pown(c, 2));
}

// Matches the gaussian (y and x) values ​​(Gauss functions) given,
// i.e. I find the coefficients a, b and c of the function
//
// f (x) = a * exp (- 0.5 * (x - b) ^ 2 / c ^ 2)
//
// for which the function best approximates the set values.
//
// The data is arranged in columns in ys and xs matrices.
//
// Kernel uses the Levenberg-Marquardt method for a non-linear problem
// least squares.
// Wikipedia: http://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
//
// Definitions needed
#define MAX_ALFA 1.0e+10
#define MIN_ALFA 1.0e-7
#define ALFA_CHANGE 10
//
__kernel void fit_gaussian
	(
		__global double * ys,	// Values ​​we adjust
		__global double * xs,	// Argumenty (x'y) for which we match
		uint width,
		__constant uint * cols_sizes, 	// Array of the number of significant elements
					   	// from the next columns of the ys matrix
		uint max_iterations,
		__global double4 * results	// Matching results,
						// at the beginning there are initial ones
						// values ​​for the coefficients a, b and c
	)
{
	uint gid = get_global_id(0);		
	if(gid >= width)
	{
		return;
	}

	// In the GN method iteratively dfrom vector we come to the "best"
	// coefficients by solving the equation each time
	// (J ^ T * J + alpha * diag (JT * J)) * dB = J ^ T * R for dB
	//
	// (Equation dB = (J ^ T * J + alpha * diag (JT * J) can be solved) ^ - 1 * J ^ T * R
	// but inverting the matrix is ​​problematic, and here they are only
	// 3 parameters so it's easier to use Cramer's formulas.)
	//
	// B - coefficients f (x)
	// dB - change of coefficient value.
	// R - differences y - f (x, B)
	// J - Jacobian for function f (x)

	// J^T * J
	real_t JTJ[3][3];
	// coefficients
   	real4_t B;
	// J^T * R	
    	real_t JTR[3]; 

	// Initial coefficient values
	// obtained by "sophisticated guess" or common sense.
	{		
		//#if defined(cl_khr_fp64)
			//B = convert_double4(results[gid]);
		//#else
			B = results[gid];
		//#endif
	}
	// Flag that match did not take place
	B.w = 0.0;

	// Index
	uint idx = gid;
	uint col_height = cols_sizes[gid];
	if(col_height == 0)
	{
		return;
	}
	uint idx_max = idx + col_height * width;

	//
	real_t alfa = 100;

	// Adjustment for current coefficients
	real_t chisq_B = 0.0;
	// Adjustment for new coefficients
	real_t chisq_newB = 0.0;
	
	for(uint i = 0; i < max_iterations; i++)
	{		
		// Data reset
		JTJ[0][0] = 0;
		JTJ[0][1] = 0;
		JTJ[0][2] = 0;
		JTJ[1][0] = 0;
		JTJ[1][1] = 0;
		JTJ[1][2] = 0;
		JTJ[2][0] = 0;
		JTJ[2][1] = 0;
		JTJ[2][2] = 0;

		JTR[0] = 0;
		JTR[1] = 0;
		JTR[2] = 0;

		chisq_B = 0.0;

		idx = gid;
		while(idx < idx_max)
		{
			real_t x = xs[idx];

			// dfda
			real_t dfda_x = dfda(x, B.y, B.z);
			//jacobian[idx] = dfd_;
			// dfdb
			real_t dfdb_x = dfdb(x, B.x, B.y, B.z);
			//jacobian[idx + width] = dfd_;
			// dfdc
			real_t dfdc_x = dfdc(x, B.x, B.y, B.z);
			//jacobian[idx + 2 * width] = dfd_

			real_t y = ys[idx];
		
        		JTJ[0][0] += dfda_x * dfda_x;
        		JTJ[0][1] += dfda_x * dfdb_x;
        		JTJ[0][2] += dfda_x * dfdc_x;

        		JTJ[1][0] += dfdb_x * dfda_x;
        		JTJ[1][1] += dfdb_x * dfdb_x;
        		JTJ[1][2] += dfdb_x * dfdc_x;

        		JTJ[2][0] += dfdc_x * dfda_x;
        		JTJ[2][1] += dfdc_x * dfdb_x;
        		JTJ[2][2] += dfdc_x * dfdc_x;

       			// R[idx]
			real_t r = y - f(x, B.x, B.y, B.z);
			// chisq_B
			chisq_B += pown(r, 2);	
			//JT * R
        		JTR[0] += dfda_x * r;
        		JTR[1] += dfdb_x * r;
        		JTR[2] += dfdc_x * r;   

			idx += width;  
		}
		chisq_B /= 2.0;

		real_t diagJTJ[3];
		diagJTJ[0] = JTJ[0][0];
		diagJTJ[1] = JTJ[1][1];
		diagJTJ[2] = JTJ[2][2];

		// The largest decline method in LM
		// (modification to the Gauss Newton algorithm)
		JTJ[0][0] += alfa * diagJTJ[0];
		JTJ[1][1] += alfa * diagJTJ[1];
		JTJ[2][2] += alfa * diagJTJ[2];

    		// (JT * J + alfa * diag(JT * J) ) * dB = JT * R is a type equation Ax = b
    		// A = (JT * J), dB = x, JT * R = b
    		// Solution with Cramer patterns, wikipedia: http://en.wikipedia.org/wiki/Cramer%27s_rule
    		// x_i = det(A_i)/det(A)

    		real_t detA = 
   		JTJ[0][0] * (JTJ[1][1] * JTJ[2][2] - JTJ[1][2] * JTJ[2][1]) -
    		JTJ[0][1] * (JTJ[1][0] * JTJ[2][2] - JTJ[1][2] * JTJ[2][0]) + 
    		JTJ[0][2] * (JTJ[1][0] * JTJ[2][1] - JTJ[1][1] * JTJ[2][0]) ;

    		real_t detA1 =
    		JTR[0]	  * (JTJ[1][1] * JTJ[2][2] - JTJ[1][2] * JTJ[2][1]) -
    		JTJ[0][1] * (  JTR[1]  * JTJ[2][2] - JTJ[1][2] * JTR[2]   ) + 
    		JTJ[0][2] * (  JTR[1]  * JTJ[2][1] - JTJ[1][1] * JTR[2]   ) ;

    		real_t detA2 = 
    		JTJ[0][0] * (JTR[1]    * JTJ[2][2] - JTJ[1][2] * JTR[2]   ) -
    		JTR[0]	  * (JTJ[1][0] * JTJ[2][2] - JTJ[1][2] * JTJ[2][0]) + 
    		JTJ[0][2] * (JTJ[1][0] * JTR[2]    - JTR[1]    * JTJ[2][0]) ;

    		real_t detA3 = 
    		JTJ[0][0] * (JTJ[1][1] * JTR[2]    - JTR[1]    * JTJ[2][1]) -
    		JTJ[0][1] * (JTJ[1][0] * JTR[2]    - JTR[1]    * JTJ[2][0]) + 
    		JTR[0]	  * (JTJ[1][0] * JTJ[2][1] - JTJ[1][1] * JTJ[2][0]) ;

		if(fabs(detA) < REAL_MIN)
		{			
			break;
		}				

		// Changing and checking stop conditions
		{    		
			real4_t dB = (real4_t)(detA1/detA, detA2/detA, detA3/detA, 0.0f);
			
			// B(k+1) = B(k) + dB	
			real4_t newB = B + dB;

					
			// Calculation of the fit for new coefficients
			// if the first change occurs.
			chisq_newB = 0.0;						
			idx = gid;			
			while(idx < idx_max)
			{
				real_t x = xs[idx];
				real_t fx = f(x, newB.x, newB.y, newB.z);
				real_t y = ys[idx];
			
				// 
				chisq_newB += pown(y - fx, 2);
				idx += width;  
			}
			chisq_newB /= 2.0;
			
			// Checking if the new coefficients are better					
			if(chisq_newB < chisq_B)
			{
				// B(k+1) = B(k)+ dB	
    				B = newB;

				// Modification towards the Gauss-Newtonwa method
				alfa = max(alfa/ALFA_CHANGE, MIN_ALFA);	
			}
			else
			{
				// We increase the share of the largest decline method
				// until we reach the maximum impact.
				while(alfa != MAX_ALFA && i < max_iterations)
				{
					i++;

					// Modification towards the largest drop method
					alfa = min(alfa*ALFA_CHANGE, MAX_ALFA);	

					// The largest decline method in LM
					// (modification to the Gauss Newton algorithm)
					JTJ[0][0] += (alfa - alfa/ALFA_CHANGE) * diagJTJ[0];
					JTJ[1][1] += (alfa - alfa/ALFA_CHANGE) * diagJTJ[1];
					JTJ[2][2] += (alfa - alfa/ALFA_CHANGE) * diagJTJ[2];

    					detA = 
   					JTJ[0][0] * (JTJ[1][1] * JTJ[2][2] - JTJ[1][2] * JTJ[2][1]) -
    					JTJ[0][1] * (JTJ[1][0] * JTJ[2][2] - JTJ[1][2] * JTJ[2][0]) + 
    					JTJ[0][2] * (JTJ[1][0] * JTJ[2][1] - JTJ[1][1] * JTJ[2][0]) ;

    					detA1 =
    					JTR[0]	  * (JTJ[1][1] * JTJ[2][2] - JTJ[1][2] * JTJ[2][1]) -
    					JTJ[0][1] * (  JTR[1]  * JTJ[2][2] - JTJ[1][2] * JTR[2]   ) + 
    					JTJ[0][2] * (  JTR[1]  * JTJ[2][1] - JTJ[1][1] * JTR[2]   ) ;

    					detA2 = 
    					JTJ[0][0] * (JTR[1]    * JTJ[2][2] - JTJ[1][2] * JTR[2]   ) -
    					JTR[0]	  * (JTJ[1][0] * JTJ[2][2] - JTJ[1][2] * JTJ[2][0]) + 
    					JTJ[0][2] * (JTJ[1][0] * JTR[2]    - JTR[1]    * JTJ[2][0]) ;

    					detA3 = 
    					JTJ[0][0] * (JTJ[1][1] * JTR[2]    - JTR[1]    * JTJ[2][1]) -
    					JTJ[0][1] * (JTJ[1][0] * JTR[2]    - JTR[1]    * JTJ[2][0]) + 
    					JTR[0]	  * (JTJ[1][0] * JTJ[2][1] - JTJ[1][1] * JTJ[2][0]) ;

					if(fabs(detA) < REAL_MIN)
					{			
						break;
					}	

					dB = (real4_t)(detA1/detA, detA2/detA, detA3/detA, 0.0);
			
					// B(k+1) = B(k) + dB	
					newB = B + dB;

					
					// Calculation of the fit for new coefficients
					// if the first change occurs.
					chisq_newB = 0.0;						
					idx = gid;			
					while(idx < idx_max)
					{
						real_t x = xs[idx];
						real_t fx = f(x, newB.x, newB.y, newB.z);
						real_t y = ys[idx];
			
						// 
						chisq_newB += pown(y - fx, 2);
						idx += width;  
					}
					chisq_newB /= 2.0;	

					if(chisq_newB < chisq_B)
					{
						// B(k+1) = B(k)+ dB	
    						B = newB;

						// Modification towards the Gauss-Newtonwa method
						alfa = max(alfa/ALFA_CHANGE, MIN_ALFA);	
						break;
					}					
				}
				
				// A better result could not be achieved for
				// the largest alpha, so we finish the whole algorithm.
				if(alfa == MAX_ALFA)
				{
					break;
				}									
			}
			
		}
	};

	// Saving the flag that the match was successful.
	B.w = 1.0;
	// Saving results	
	//results[gid] = convert_double4(B);
	results[gid] = B;
}

//
// Calculates Gauss functions, where xs is a matrix with function arguments.
//
__kernel void calc_gaussian
	(
		__global double * xs,	// Arguments for which we calculate gaussian
		__global double4 * gparams,	// Coefficients of the Gauss function
		uint width,		
		__constant uint * cols_sizes, 	// An array of the number of significant elements
					   	// from the next columns of the xs matrix		
		__global double * ys	// Function results
	)
{
	uint gid0 = get_global_id(0);		
	if(gid0 >= width)
	{
		return;
	}

	// Number of items in the column
	uint col_height = cols_sizes[gid0];

	uint gid1 = get_global_id(1);	
	if(gid1 >= col_height)
	{
		return;
	}

	// Index
	uint idx = gid0 + width * gid1;	
	
	//Get x
	real_t x = xs[idx];

	// Get parameters for this column of x's
	__local double4 abc_local;
	real4_t abc;
	if(gid1 == 0)
	{
		abc_local = gparams[gid0];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	#if defined(cl_khr_fp64)
		abc = convert_double4(abc_local);
	#else
		abc = abc_local;
	#endif

	// Calculation
	real_t fx = abc.x * exp( (real_t)(-0.5) * pown((x - abc.y), 2) / pown(abc.z, 2));

	// Record
	ys[idx] = convert_double(fx);
}

//
// 
//
__kernel void calc_gaussian_chisq
	(
		__global double * xs,	// Arguments for which we calculate gaussian
		__global double * ys,	// 
		__global double * errors,	// 
		__global double4 * gparams,	// Coefficients of the Gauss function
		uint width,		
		__constant uint * cols_sizes, 	// from the next columns of the xs matrix
					   	// 		
		__global double * chisqs	// Chi square results
	)
{
	uint gid = get_global_id(0);		
	if(gid >= width)
	{
		return;
	}

	// Number of items in the column
	uint col_height = cols_sizes[gid];

	// Index
	uint idx = gid;	
	uint idx_max = gid + col_height * width;

	real4_t abc;
	#if defined(cl_khr_fp64)
		abc = convert_double4(gparams[gid]);
	#else
		abc = gparams[gid];
	#endif
	
	real_t chisq = (real_t)(0.0);
	real_t c = (real_t)(0.0);
	while (idx < idx_max) 
	{
		real_t f, y, e, t, u, x;

		y = ys[idx];
		x = xs[idx];
		f = abc.x * exp( (real_t)(-0.5) * pown((x - abc.y), 2) / pown(abc.z, 2));
		e = errors[idx];

		u = pown((y-f),2) / pown(e, 2) - c;
		t = chisq + u;
		c = (t - chisq) - u;

		chisq = t;

		idx += width;
	}

	// Record
	chisqs[gid] = convert_double(chisq);
}

#define C (real_t)(299792458.0)

//
// 
//
__kernel void calc_gaussian_fwhm
	(
		__global double4 * gparams,	// Coefficients of the Gauss function
		__global double * fwhms,	// 
		uint width
	)
{
	uint idx = get_global_id(0);		
	if(idx >= width)
	{
		return;
	}

	real4_t abc;
	#if defined(cl_khr_fp64)
		abc = convert_double4(gparams[idx]);
	#else
		abc = gparams[idx];
	#endif

	real_t b = abc.y;
	real_t c = abc.z;

	//conversion of the unit of the c coefficient (sigma): A -> km / s
	real_t c_kms = C;
 	c_kms *= (pown(((real_t)(1.0) + c/b), 2) - (real_t)(1.0));
	c_kms /= (pown(((real_t)(1.0) + c/b), 2) + (real_t)(1.0));
	c_kms *= (real_t)(1.0e-3);
	
	// c (sigma) -> FWHM
	real_t fwhm = c_kms * (real_t)(2.0);
	fwhm *= pow( ( (real_t)(2.0) * log((real_t)(2.0)) ), (real_t)(0.5) );

/*
	// Yes it is in the script, but the comments in the same script show
	// that this is a bad version.

	c = c * (real_t)(2.0);
	c *= pow( ( (real_t)(2.0) * log((real_t)(2.0)) ), (real_t)(0.5) );

	real_t fwhm = C;
 	fwhm *= (pown(((real_t)(1.0) + c/b), 2) - (real_t)(1.0));
	fwhm /= (pown(((real_t)(1.0) + c/b), 2) + (real_t)(1.0));
	fwhm *= (real_t)(1.0e-3);
*/	

	// Zapis
	fwhms[idx] = convert_double(fwhm);
}

#if defined(cl_khr_fp64)
	#pragma OPENCL EXTENSION cl_khr_fp64 : disable
#endif
