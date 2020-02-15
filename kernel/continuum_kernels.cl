#define ASTRO_OBJ_SPEC_SIZE 4096

__kernel void fix_reglin_results
	(
		__global double8 * c_reglin_results, //  Simple linear regression parameters cont
		__global double8 * reglin_results, // Simple linear regression parameters
		uint size	// Number of parameters
	)
{
	// gid0
	uint gid0 = get_global_id(0);

	if(gid0 >= size)
	{
		return;
	}
    printf("%d", gid0);	
	// r =  (a, b, sia2, sib2, siy2, x_sum, x^2_sum, y_sum)
	double8 r = reglin_results[gid0];	

	// b = 10^b
	r.s1 = pow(10.0, r.s1);
	// sia2 = sia2^0.5
	r.s2 = pow(r.s2, 0.5);
	// sib2 = 10^b * log(10) * sib2^0.5	
	r.s3 = r.s1 * log(10.0) * pow(r.s3, 0.5);
	// siy2 = siy2^0.5
	r.s4 = pow(r.s4, 0.5);

	reglin_results[gid0] = r;

	r = c_reglin_results[gid0];
	r.s1 = pow(10.0, r.s1);
	c_reglin_results[gid0] = r;
}

__kernel void calc_cfun_dcfun
	(
		__global double * wavelengths_matrix, 	// Wavelengths for spectra [original]
		__global double * dcfuns_matrix, 	// 
		__global double * cfuns_matrix, // Continuum
		__global double8 * reglin_results, 	// Simple linear regression parameters
		__global double8 * c_reglin_results 	
	)
{
	uint gid0 = get_global_id(0);
	uint gid1 = get_global_id(1);	

	// Calculation of the element index
	uint idx = gid0 * ASTRO_OBJ_SPEC_SIZE + gid1;

	uint col_idx = idx % get_global_size(0);  

	//         (0,    1,    2,    3,    4, ....)
	// cpar =  (a, 10^b, sia2, sib2, siy2, ...)
	double8 cpar = reglin_results[col_idx];

    // printf("col_idx %f", cpar);

	double wavelength = wavelengths_matrix[idx];

	double cfun = cpar.s1 * pow(wavelength, cpar.s0);
	cfuns_matrix[idx] = cfun;
		
	//        (0, 1,           2,                         3,        4, ...)
	// par =  (a, 10^b, sia2^0.5, 10^b * log(10) * sib2^0.5, siy2^0.5, ...)
	double8 par = c_reglin_results[col_idx];

	// dcfun= sib * la^a + sia * b * a * log(lambda)
	double dcfun = par.s3 * pow(wavelength, par.s0);
	dcfun += par.s2 * par.s1 * par.s0 * log(wavelength);
	dcfuns_matrix[idx] = dcfun;
}

__kernel void calc_cw
	(
		__global double * wavelengths_matrix_filtered, 	// Wavelengths for spectra after filtration
		__global double * cfuns_filtered,	// Continuum for elements after filtration (in windows)
		uint filtered_size,			// The maximum number of significant elements after filtration
		__global double8 * c_reglin_results 	// Simple linear regression parameters
	)
{
	uint gid0 = get_global_id(0);
	uint gid1 = get_global_id(1);	

	//Calculation of the element index
	uint idx = gid0 * filtered_size + gid1;
	uint col_idx = idx % get_global_size(0);  

	//         (0,    1,    2,    3, ....)
	// cpar =  (a, 10^b, sia2, sib2, ...)
	double4 cpar = c_reglin_results[col_idx].lo;
	
	double wavelength = wavelengths_matrix_filtered[idx];

	double cfun_filtered = cpar.s1 * pow(wavelength, cpar.s0);
	cfuns_filtered[idx] = cfun_filtered;			
}

__kernel void reduce_continuum_chisqs
	(
		__global double * chisqs, 		// Buffer with chisq
		__constant uint * filtered_sizes, 	// Number of significant elements after filtration
		uint size				// The amount of chisq and filtered_sizes
	)
{
	// gid0 - item number from chisqs (i.e. one chisq)
	uint gid0 = get_global_id(0);

	if(gid0 >= size)
	{
		return;
	}
		
	double filtered_size, chisq;

	filtered_size = (double)filtered_sizes[gid0];
	chisq = chisqs[gid0];

	chisq /= filtered_size;

	chisqs[gid0] = chisq;	
}
