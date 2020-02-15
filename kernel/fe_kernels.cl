
__kernel void reduce_fe_chisqs
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

	chisq /= (filtered_size - 1);

	chisqs[gid0] = chisq;	
}
