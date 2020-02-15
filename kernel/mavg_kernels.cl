
__kernel void simple_mavg
	(
		__global double * input,
		uint width,
		uint height,
		__global double * output,
		uint window_width
	)
{
	// column 
	uint gid = get_global_id(0);
	uint idx = gid;

	if(gid >= width)
	{
		return;
	}
		
	unsigned long end = idx + height * width;
	uint i = 0;
	double lastSum = 0;
	double result = 0;

	while(i < window_width && idx < end)
	{
		lastSum += input[idx];		
		result = lastSum / ((double)(i+1));
		output[idx] = result;
		
		idx += width;	
		i++;
	}
	
	double fwindow_width = (double)(window_width);
	while(idx < end)
	{
		double new = input[idx];
		double old = input[idx - (window_width*width)];

		lastSum = lastSum - old + new;	
		result = lastSum / fwindow_width;
		output[idx] = result;

		idx += width;	
	}
    // printf("gid %d %f %f", gid, input[gid], output[gid]);
    // printf("h w %d %d indeksy %d %d wektory %d %d matrix %f ", a_height, a_width, get_global_id(0), get_global_id(1), read_idx, write_idx, a[read_idx]);
	return;
}

__kernel void centered_mavg
	(
		__global double * input,
		uint width,
		uint height,
		__constant uint * cols_heights,
		__global double * output,
		uint window_width
	)
{
	// column 
	uint gid = get_global_id(0);
	uint idx = gid;

	if(gid >= width)
	{
		return;
	}

	uint col_height = cols_heights[gid];
		
	unsigned long end = idx + col_height * width;
	uint i = 0;
	double lastSum = 0;
	double result = 0;

	//
	// For the first window_width - 1 elements [0; window_width - 1)
	//

	// Moving average value for the first window_width of the elements is
	// calculated so that for the i-th element is equal to the arithmetic mean
	// from element with index 0 to element with index i + window_width
	while(i < window_width && idx < end)
	{
		lastSum += input[idx];			
		idx += width;	
		i++;
	}

	i = 0;
	while(i < window_width && idx < end)
	{	
		result = lastSum / ((double)(window_width + i));
		output[idx - (window_width * width)] = result;
		
		double new = input[idx];
		lastSum = lastSum + new;

		idx += width;	
		i++;
	}

	//
	// For items with an index in the range [window_width; number_elements - window_width]
	//

	// The moving average value for the ith element is the arithmetic mean of the elements
	// indexes in the range (i - window_width; and + window_width)
	// idx = gid;
	// lastSum = 0;
	// i = 0;
	// while (i <2 * window_width && idx <end)
	// {
	// lastSum + = input [idx];
	// idx + = width;
	// i ++;
	//}
	
	double fwindow_width = (double)(2 * window_width);
	while(idx < end)
	{			
		result = lastSum / fwindow_width;
		output[idx - (window_width * width)] = result;

		double new = input[idx];
		double old = input[idx - (2 * window_width * width)];
		lastSum = lastSum - old + new;	

		idx += width;	
	}
	//
	// For items with an index in the range (number_elements - window_width; number_elements]
	//

	// The moving average value for the ith element is the arithmetic mean of the elements
	// about indexes in the range (i; number_elements)

	lastSum = 0.0f;
	idx -= 2 * window_width * width;
	while(idx < end)
	{		
		lastSum += input[idx];
		idx += width;	
	}

	idx -= window_width * width;	
	i = 2 * window_width;
	while(idx < end)
	{			
		result = lastSum / ((double)(i));
		output[idx] = result;
		
		double old = input[idx - window_width * width];
		lastSum = lastSum - old;	

		idx += width;	
		i--;
	}

    // printf("gid %d %f %f", gid, input[gid], output[gid]);
	return;
}
