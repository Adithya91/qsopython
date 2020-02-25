
import pyopencl as cl
import numpy as np
MAX_FITGAUSSIAN_LM_ITERS = 500



def calcGlobalSize(workGroupMultiple, dataSize):
    size = dataSize
    remainder = size % workGroupMultiple
    if (remainder != 0):
        size += workGroupMultiple - remainder
    if (size < dataSize):
        print("Error in calculating global_work_size.")
    return size

#To fit gaussian parameters to the quasar spectrum
def fitGaussian(QsoCL,yMatrix, xMatrix, sizesVector, resultsVector):
 queue = cl.CommandQueue(QsoCL.ctx)
 
 h = yMatrix.shape[0]  # spectrumsSize
 w = yMatrix.shape[1]  # spectrumsNumber
 res = np.asarray(resultsVector,dtype=np.float64)
 y_g = QsoCL.readBuffer(yMatrix)
 x_g = QsoCL.readBuffer(xMatrix)
 res_g = QsoCL.makeBuffer(res)
 sizes_g = QsoCL.readBuffer(sizesVector)
 
 _knl = QsoCL.buildKernel('gaussian_kernels.cl').fit_gaussian
 _knl._wg_info_cache = {}
 workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
 globalSize = calcGlobalSize(workGroupMultiple, w)
 _knl.set_scalar_arg_dtypes(
        [None, None,np.uint32, None, np.uint32,None])
 _knl(queue, (globalSize,),None,
         y_g, x_g,w,sizes_g,MAX_FITGAUSSIAN_LM_ITERS, res_g)
 cl.enqueue_copy(queue, res, res_g)
 return res


#To calculate the gaussian parameters
def calcGaussian(QsoCL,xMatrix, gaussianParamsVector, sizesVector):
 queue = cl.CommandQueue(QsoCL.ctx)
 
 h = xMatrix.shape[0]  # spectrumsSize
 w = xMatrix.shape[1]  # spectrumsNumber
 fxs = np.zeros((w,h),dtype=np.float64)
 fxs = np.transpose(np.asarray(fxs, dtype=np.float64, order='F'))
 x_g = QsoCL.readBuffer(xMatrix)
 gaussianParams_g = QsoCL.readBuffer(gaussianParamsVector)
 sizes_g = QsoCL.readBuffer(sizesVector)

 res_g = QsoCL.writeBuffer(fxs)
 _knl = QsoCL.buildKernel('gaussian_kernels.cl').calc_gaussian
 _knl._wg_info_cache = {}
 workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
 globalSize = QsoCL.calcGlobalSize(h)
 _knl.set_scalar_arg_dtypes(
        [None, None,np.uint32, None,None])
 _knl(queue, (w,globalSize),(1,workGroupMultiple),
         x_g, gaussianParams_g,w,sizes_g, res_g)
 cl.enqueue_copy(queue, fxs, res_g)
 return fxs



#To perform chi-squared test
def calcGaussianChisqs(QsoCL,xMatrix, yMatrix, errorsMatrix, gaussianParamsVector, 
		      sizesVector):
 queue = cl.CommandQueue(QsoCL.ctx)
 
 h = xMatrix.shape[0]  # spectrumsSize
 w = xMatrix.shape[1]  # spectrumsNumber
 results = np.zeros(w,dtype=np.float64)
 x_g = QsoCL.readBuffer(xMatrix)
 y_g = QsoCL.readBuffer(yMatrix)
 err_g = QsoCL.readBuffer(errorsMatrix)
 gaussianParams_g = QsoCL.readBuffer(gaussianParamsVector)
 sizes_g = QsoCL.readBuffer(sizesVector)

 res_g = QsoCL.writeBuffer(results)
 _knl = QsoCL.buildKernel('gaussian_kernels.cl').calc_gaussian_chisq
 _knl._wg_info_cache = {}
 workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
 globalSize = calcGlobalSize(workGroupMultiple,w)
 _knl.set_scalar_arg_dtypes(
        [None, None,None,None,np.uint32, None,None])
 _knl(queue, (globalSize,),(workGroupMultiple,),
         x_g,y_g,err_g,gaussianParams_g,w,sizes_g, res_g)
 cl.enqueue_copy(queue, results, res_g)
 return results



#To determine the full width half maximum i.e. the difference between two 
#extreme values of the independent variable at which the dependent variable 
#is equal t half of its maximum value.
def calcGaussianFWHM(QsoCL,gaussianParamsVector):
 queue = cl.CommandQueue(QsoCL.ctx)
 
 size = len(gaussianParamsVector) 
 gaussianFWHMs = np.zeros(size,dtype=np.float64)
 globalSize = QsoCL.calcGlobalSize(size)
 gaussianParams_g = QsoCL.readBuffer(gaussianParamsVector)
 
 res_g = QsoCL.writeBuffer(gaussianFWHMs)
 _knl = QsoCL.buildKernel('gaussian_kernels.cl').calc_gaussian_fwhm
 _knl._wg_info_cache = {}
 workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
 _knl.set_scalar_arg_dtypes(
        [None, None,np.uint32])
 _knl(queue, (globalSize,),(workGroupMultiple,),
         gaussianParams_g,res_g,size)
 cl.enqueue_copy(queue, gaussianFWHMs, res_g)
 return gaussianFWHMs
