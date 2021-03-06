#!/usr/bin/env python

import pyopencl as cl
import numpy as np
from QsoCL import *
MAX_FITGAUSSIAN_LM_ITERS = 500
ASTRO_OBJ_SPEC_SIZE = 4096



def calcGlobalSize(workGroupMultiple, dataSize):
    size = dataSize
    remainder = size % workGroupMultiple
    if (remainder != 0):
        size += workGroupMultiple - remainder
    if (size < dataSize):
        print("Error in calculating global_work_size.")
    return size

#To create an output buffer object and host variable
def outputMatrixTran(QsoCL,w,h):
    outMat=np.zeros((w,h), dtype=np.float64)
    outMat = np.transpose(np.asarray(outMat, dtype=np.float64, order='F'))   
    out_g = QsoCL.writeBuffer(outMat)
    return outMat,out_g

#To return matrices after filtering windows
def filter_matrix_fit(QsoCL,
        spectrumsLinesMatrix,
        wavelengthsMatrix,
        continuumsMatrix,
        sizes,element_range_list,
        h,w,elemSize):


    queue = cl.CommandQueue(QsoCL.ctx)    
    
    _knl = QsoCL.buildKernel('spectrums_kernels.cl').filterWithWavelengthWindows
    _knl.set_scalar_arg_dtypes(
        [None, None, None, None, None, np.uint32])
    _knl(queue, (w, ASTRO_OBJ_SPEC_SIZE), (1, QsoCL.maxWorkGroupSize),
         wavelengthsMatrix,spectrumsLinesMatrix,continuumsMatrix, sizes, element_range_list, elemSize)
    
    sp = spectrumsLinesMatrix.int_ptr
    wv = wavelengthsMatrix.int_ptr
    cont = continuumsMatrix.int_ptr
    
    ###
    newSizes = np.zeros(w, dtype=np.int32)
    sizes_g = QsoCL.writeBuffer(newSizes)
    _knl = QsoCL.buildKernel('tools_kernels.cl').countIfNotInf
    _knl._wg_info_cache = {}
    workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
    globalsize = calcGlobalSize(QsoCL.maxWorkGroupSize, w)
    _knl.set_scalar_arg_dtypes([None, np.uint32, np.uint32, None])
    _knl(queue, (globalsize,),
         (workGroupMultiple,), cl.Buffer.from_int_ptr(sp), w, h, sizes_g)
    cl.enqueue_copy(queue, newSizes, sizes_g)    
    maxSize = max(newSizes)
    ###
        
    outMat,out_g = outputMatrixTran(QsoCL,w,maxSize)

    _knl = QsoCL.buildKernel('tools_kernels.cl').copyIfNotInf
    _knl._wg_info_cache = {}
    workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
    globalsize = QsoCL.calcGlobalSize(w)
    _knl.set_scalar_arg_dtypes([None, np.uint32, np.uint32, None, np.uint32])
    
    _knl(queue, (globalsize,),
         (workGroupMultiple,), cl.Buffer.from_int_ptr(sp), w, h, out_g, maxSize)
    cl.enqueue_copy(queue, outMat, out_g)
    specMat = outMat
    
    
    outMat,out_g = outputMatrixTran(QsoCL,w,maxSize)
    _knl(queue, (globalsize,),
         (workGroupMultiple,), cl.Buffer.from_int_ptr(wv), w, h, out_g, maxSize)
    cl.enqueue_copy(queue, outMat, out_g)
    wavMat = outMat
        

    outMat,out_g = outputMatrixTran(QsoCL,w,maxSize)
    _knl(queue, (globalsize,),
         (workGroupMultiple,), cl.Buffer.from_int_ptr(cont), w, h, out_g, maxSize)
    cl.enqueue_copy(queue, outMat, out_g)
    contMat = outMat    
    
    out = {
    "wavelengthsMatrix":wavMat,
    "spectrumLinesMatrix":specMat,
    "continuumsMatrix":contMat,
    "sizes":newSizes
           }
    print("filter results:",out)
    return out

#To fit gaussian parameters to the quasar spectrum
def fit_Gaussian(QsoCL,spectrumsLinesMatrix, 
             wavelengthsMatrix,continuumsMatrix, errorsMatrix,
             wavelengthsMatrixCopy,spectrumsLinesMatrixCopy,
             sizes,sizesCopy,fitGResults,h,w):
 queue = cl.CommandQueue(QsoCL.ctx)
 
 fitResults = np.asarray(fitGResults,dtype=np.float64)
 fitGResults = QsoCL.makeBuffer(np.asarray(fitGResults,dtype=np.float64))

 _knl = QsoCL.buildKernel('gaussian_kernels.cl').fit_gaussian
 _knl._wg_info_cache = {}
 workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
 globalSize = calcGlobalSize(workGroupMultiple, w)
 _knl.set_scalar_arg_dtypes(
        [None, None,np.uint32, None, np.uint32,None])
 _knl(queue, (globalSize,),None,
         spectrumsLinesMatrix, wavelengthsMatrix,w,sizes,MAX_FITGAUSSIAN_LM_ITERS, fitGResults)
 cl.enqueue_copy(queue, fitResults, fitGResults) 

 gauss,gauss_g = outputMatrixTran(QsoCL,w,h)
 _knl = QsoCL.buildKernel('gaussian_kernels.cl').calc_gaussian
 _knl._wg_info_cache = {}
 workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
 globalSize = QsoCL.calcGlobalSize(h)
 _knl.set_scalar_arg_dtypes(
        [None, None,np.uint32, None,None])
 _knl(queue, (w,globalSize),(1,workGroupMultiple),
         wavelengthsMatrix, fitGResults,w,sizes, gauss_g)
 
 
 outMat,out_g = outputMatrixTran(QsoCL,w,h)
 globalSize = QsoCL.calcGlobalSize(w)
 _knl = QsoCL.buildKernel('basics_kernels.cl').matrix_divide_matrix
 _knl(queue, (h, globalSize),
         (1, QsoCL.maxWorkGroupSize), gauss_g, np.uint32(w),continuumsMatrix,out_g)
 
 ews = np.zeros((h,w), dtype=np.float64)
 ews_g = QsoCL.writeBuffer(ews)
 _knl = QsoCL.buildKernel('tools_kernels.cl').integrate_trapz
 _knl._wg_info_cache = {}
 workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
 _knl(queue, (globalSize,),
         (workGroupMultiple,), out_g, wavelengthsMatrix,np.uint32(w),np.uint32(h),sizes,ews_g)
 cl.enqueue_copy(queue, ews, ews_g) 
 
 gaussianChisqs = np.zeros(w,dtype=np.float64)
 chisqs_g = QsoCL.makeBuffer(gaussianChisqs)
 _knl = QsoCL.buildKernel('gaussian_kernels.cl').calc_gaussian_chisq
 _knl._wg_info_cache = {}
 workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
 globalSize = calcGlobalSize(workGroupMultiple,w)
 _knl.set_scalar_arg_dtypes(
        [None, None,None,None,np.uint32, None,None])
 _knl(queue, (globalSize,),(workGroupMultiple,),
         wavelengthsMatrixCopy,spectrumsLinesMatrixCopy,errorsMatrix,fitGResults,w,sizesCopy, chisqs_g)
 cl.enqueue_copy(queue, gaussianChisqs, chisqs_g)
 
 size = len(fitResults) 
 gaussianFWHMs = np.zeros(size,dtype=np.float64)
 fwhms_g = QsoCL.writeBuffer(gaussianFWHMs)
 globalSize = QsoCL.calcGlobalSize(size)
 _knl = QsoCL.buildKernel('gaussian_kernels.cl').calc_gaussian_fwhm
 _knl._wg_info_cache = {}
 workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
 _knl.set_scalar_arg_dtypes(
        [None, None,np.uint32])
 _knl(queue, (globalSize,),(workGroupMultiple,),
         fitGResults,fwhms_g,size)
 cl.enqueue_copy(queue, gaussianFWHMs, fwhms_g)
 
 elDict = {"fitResults": fitResults,
	    "ews" : ews,
	    "chisqs":gaussianChisqs,
	    "gaussianFWHMs": gaussianFWHMs
            }
 return elDict

  
