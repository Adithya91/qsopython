#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define ASTRO_OBJ_SPEC_SIZE 4096

__kernel void generateWavelengthsMatrix
	(
		__global double4 * abz_buffer,
		__global double * wavelengths_matrix
	)
{
	// gid0 - Quasar spectrum number
	uint gid0 = get_global_id(0);	
	
	// parameters a, b and quasar
	double4 abz_;
	__local double4 local_abz_;

	if (get_local_id(0) == 0)
	{
  		local_abz_ = abz_buffer[gid0];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	abz_ = local_abz_;
	
	// gid1 - spectrum element number (index from 0 to 4095)
	uint gid1 = get_global_id(1);
	uint idx = gid0 * get_global_size(1) + gid1;
	
	// Lambda calculation for this gid1
	double wl = (abz_.x * (double)(gid1)) + abz_.y;
	wl = pow((double)(10),wl);
	
	// Taking into account the cosmological shift - transition to a system emitting using redshift
	wavelengths_matrix[idx] = wl / (abz_.z + (double)(1));
	
	return;
}


// Adds one spectrum to each of the spectra in spectrums_matrix
// with single interpolation.
//
// Spectra, spectral lengths are set here by columns, not by rows.
// thanks to that we get a big leap in performance.
//
__kernel void addSpectrum
	(
		__global double * wavelengths_matrix,	// Lengths for spectra
		__global double * spectrums_matrix,	// Column spectra
		__global uint  * sizes,			// Sizes of the spectra
		uint size,				// Number of spectra (number of columns)
		__constant double * to_add_wavelengths,	// The wavelengths of the spectrum that we add
		__constant double * to_add_spectrum,	// The spectrum we are adding
		uint to_add_spectrum_size,	
		__global double * output_matrix		// Matrix of the sum result record
	)
{
	// gid0 - Quasar spectrum number
	uint gid0 = get_global_id(0);
	if(gid0 >= size)
	{
		return;
	}	
	
	uint spectrum_size = sizes[gid0];

	// Element index with wavelengths_matrix / spectrums_matrix
	uint idx = gid0;
	uint idx_max = idx + spectrum_size * size;	
	
	// Element index of to_add_wavelengths / to_add_spectrum
	uint to_add_idx = 0;
	uint to_add_idx_max = to_add_spectrum_size - 1;
	
	double wavelength = wavelengths_matrix[idx];
	double value = spectrums_matrix[idx];
	
	
	
	double to_add_wl 	= to_add_wavelengths[to_add_idx];
	double to_add_wl_next 	= to_add_wavelengths[to_add_idx+1];		
	double to_add_value 	= to_add_spectrum[to_add_idx];
	double to_add_value_next = to_add_spectrum[to_add_idx+1];	
	
   //printf("gid0=%u,idx_max=%u,to_add_idx_max=%u,to_add_wl=%lf,to_add_wl_next=%lf,to_add_value=%lf,to_add_value_next=%lf,wavelength=%.3lf,value=%lf",gid0,idx_max,to_add_idx_max,to_add_wl,to_add_wl_next,to_add_value,to_add_value_next,wavelength,value);

  
	int idx_max_flag = 0;
	int to_add_idx_max_flag = 0;

	while(1)
	{				
		if(wavelength >= to_add_wl)
		{			
			if(wavelength <= to_add_wl_next)
			{	
				double a = (to_add_value_next - to_add_value)/(to_add_wl_next - to_add_wl);
				double b = to_add_value - (a * to_add_wl);
				value = value + (a * wavelength + b);		
				output_matrix[idx] = value;		
				
				idx += size;
				// Checking if idx did not exceed idx_max
				// Before we read the data.
				idx_max_flag = select(1, 0, idx < idx_max);
				if(idx_max_flag)
				{
					break;
				}
				wavelength = wavelengths_matrix[idx];
				value = spectrums_matrix[idx];
			}
			else
			{
				to_add_idx++;
				// Check if to_add_idx did not exceed to_add_idx_max
				// Before we read the data.
				to_add_idx_max_flag = select(1, 0, to_add_idx < to_add_idx_max);
				if(to_add_idx_max_flag)
				{
					break;
				}
				to_add_wl = to_add_wl_next;				
				to_add_wl_next = to_add_wavelengths[to_add_idx+1];
				
				to_add_value = to_add_value_next;
				to_add_value_next = to_add_spectrum[to_add_idx+1];
			}
		}
		else
		{	
			output_matrix[idx] = value;
		
			idx += size;
			// Checking if idx did not exceed idx_max
			// Before we read the data.
			idx_max_flag = select(1, 0, idx < idx_max);
			if(idx_max_flag)
			{
				break;
			}
			wavelength = wavelengths_matrix[idx];
			value = spectrums_matrix[idx];
		}	
	}
	
	while(idx < idx_max)
	{
		value = spectrums_matrix[idx];
		output_matrix[idx] = value; 
		idx += size;
	}	
}




// Filters only data whose corresponding value
// the wavelength of wavelengths_matrix is ​​in some spectral window.
//
__kernel void filterWithWavelengthWindows
	(
		__global double * wavelengths_matrix,	// Spectral wavelengths
		__global double * spectrums_matrix,	// spectrum
		__global double * errors_matrix,		// Spectrum measurement errors
		__global uint  * sizes,		 	//Spectra sizes in spectrums_matrix
		__constant double2 * windows,	// Wavelength windows
		uint windows_size 		// Number of windows
	)	
{
	
	uint gid0 = get_global_id(0);
	
	uint gid1 = get_global_id(1);
	
	uint idx = (gid0 * ASTRO_OBJ_SPEC_SIZE) + gid1;

	uint size = sizes[idx % get_global_size(0)];

	// Zero indicates a lack of matching to any window
	int global_flag = 0;

	double wl_result = wavelengths_matrix[idx];
	double spec_result = spectrums_matrix[idx];
	double error_result = errors_matrix[idx];

	uint window_idx = 0;
	double2 window;
	__local double2 window_local;
	
	// Loop through all windows
	int window_flag;
	for(; window_idx < windows_size; window_idx++)
	{
		window_flag = 0;
		if (get_local_id(0) == 0)
		{
  			window_local = windows[window_idx];
		}
		//barrier(CLK_LOCAL_MEM_FENCE);

		window = window_local;

		window_flag = select(0, 1, wl_result >= window.x);
		window_flag *= select(0, 1, wl_result <= window.y);
		// If both conditions are met, we have matches
		// Otherwise, we leave as it is.

		global_flag = select(global_flag, 1, window_flag == 1);
	}
	
	spec_result = select((double)INFINITY, spec_result, (long)(global_flag == 1));
	wl_result = select((double)INFINITY, wl_result, (long)(global_flag == 1));
	error_result = select((double)INFINITY, error_result, (long)(global_flag == 1));	

	uint row_idx = idx / get_global_size(0);
	spec_result = select((double)INFINITY, spec_result, (long)(row_idx < size));
	wl_result = select((double)INFINITY, wl_result, (long)(row_idx < size));
	error_result = select((double)INFINITY, error_result, (long)(row_idx < size));

	wavelengths_matrix[idx] = wl_result;
	//spectrums_matrix[idx] = (double)(global_flag==1);
	spectrums_matrix[idx] = spec_result;
	errors_matrix[idx] = error_result;
}


__kernel void filterNonpositive
	(		
		__global double * spectrums_matrix,	// spectrum
		__global double * a_matrix,	// Any size matrix of spectrums_matrix
		__global double * b_matrix,	// 
		__constant uint * sizes 	// Sizes of spectra in spectrums_matrix
	)
{
	uint gid0 = get_global_id(0);
	uint gid1 = get_global_id(1);
	// Item index
	uint idx = (gid0 * ASTRO_OBJ_SPEC_SIZE) + gid1;
	
	uint size = sizes[idx % get_global_size(0)];
	uint row_idx = idx / get_global_size(0);

	double spectrum = spectrums_matrix[idx];

	// Flag calculation, spec_result> 0.
	// Zero indicates a number less than zero
	long flag = select(0, 1, spectrum >= (double)FLT_MIN);
	// Additional checking if the item is significant.
	flag *= select(0, 1, row_idx < size);

	spectrum = select((double)INFINITY, spectrum, flag);
	spectrums_matrix[idx] = spectrum;

	double a = a_matrix[idx];
	a = select((double)INFINITY, a, flag);
	a_matrix[idx] = a;	

	double b = b_matrix[idx];
	b = select((double)INFINITY, b, flag);	
	b_matrix[idx] = b;
}


// Filters only the data that matches
// values ​​from spectrums_matrix are different from zero.
//
__kernel void filterZeros
	(
		__global double * spectrums_matrix,	// spectrum
		__global double * a_matrix,	// Any size matrix of spectrums_matrix
		__global double * b_matrix,	// 
		__constant uint * sizes 	// Spectra sizes in spectrums_matrix
	)
{
	//line number
	uint gid0 = get_global_id(0);
	//column number
	uint gid1 = get_global_id(1);
	// Index element
	uint idx = (gid0 * ASTRO_OBJ_SPEC_SIZE) + gid1;
	
	uint size = sizes[idx % get_global_size(0)];
	uint row_idx = idx / get_global_size(0);

	double spectrum = spectrums_matrix[idx];

	// Calculation of the flag, or spec_result! = 0.
	long flag = select(0, 1, ((spectrum != 0.0) && (row_idx < size)));
	// Additional checking if the item is significant.
	//flag* = select(0, 1, row_idx < size);
	//double inf = (double)INFINITY;
	
	spectrum = select((double)INFINITY, spectrum, flag);
	spectrums_matrix[idx] = spectrum;

	double a = a_matrix[idx];
	a = select((double)INFINITY, a, flag);
	a_matrix[idx] = a;	

	double b = b_matrix[idx];
	b = select((double)INFINITY, b, flag);	
	b_matrix[idx] = b;
}


// Filters only the data that matches
// values ​​from spectrums_matrix are different from +/- INFINITY.
//

__kernel void filterInfs
	(
		__global double * spectrums_matrix,	// spectrum
		__global double * a_matrix,	// Any size matrix of spectrums_matrix
		__global double * b_matrix,	// 
		__constant uint * sizes 	// Spectra sizes in spectrums_matrix
	)
{
	uint gid0 = get_global_id(0);
	uint gid1 = get_global_id(1);
	
	uint idx = (gid0 * ASTRO_OBJ_SPEC_SIZE) + gid1;

	uint size = sizes[idx % get_global_size(0)];
	uint row_idx = idx / get_global_size(0);

	double spectrum = spectrums_matrix[idx];

	// Calculation of the flag, or spec_result! = +/- INFINITY.
	long flag = select(0, 1, spectrum != (double)INFINITY);
	flag *= select(0, 1, spectrum != -(double)INFINITY);
	// Additional checking if the item is significant.
	flag *= select(0, 1, row_idx < size);

	spectrum = select((double)INFINITY, spectrum, flag);
	spectrums_matrix[idx] = spectrum;

	double a = a_matrix[idx];
	a = select((double)INFINITY, a, flag);
	a_matrix[idx] = a;
		
	if(b_matrix != 0)
	{
		double b = b_matrix[idx];
		b = select((double)INFINITY, b, flag);	
		b_matrix[idx] = b;
	}
}
