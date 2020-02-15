#!/usr/bin/env python

import pyopencl as cl
import numpy as np
import math
from Quasarcl import *
ASTRO_OBJ_SPEC_SIZE = 4096


def calcGlobalSize(workGroupMultiple, dataSize):
    size = dataSize
    remainder = size % workGroupMultiple
    if (remainder != 0):
        size += workGroupMultiple - remainder
    if (size < dataSize):
        print("Error in calculating global_work_size.")
    return size

#To perform Chi-squared test
def chisqs(QuasarCL,specFiltered,contFiltered,errFiltered,sizes,h,w): 
    queue = cl.CommandQueue(QuasarCL.ctx)
    globalSize = QuasarCL.calcGlobalSize(w)
    output = np.zeros(w, dtype=np.double)
    out_g = QuasarCL.writeBuffer(output)
    
    _knl = QuasarCL.buildKernel('tools_kernels.cl').chisq
    _knl._wg_info_cache = {}
    workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QuasarCL.devices[0])
    _knl.set_scalar_arg_dtypes(
        [None, None, None, np.uint32, np.uint32, None, None])
    _knl(queue, (globalSize,),
         (workGroupMultiple,), specFiltered, contFiltered, errFiltered, w, h, sizes, out_g)
    cl.enqueue_copy(queue, output, out_g)
    size= len(output)  

    globalSize = QuasarCL.calcGlobalSize(size)
    _knl = QuasarCL.buildKernel('continuum_kernels.cl').reduce_continuum_chisqs

    _knl.set_scalar_arg_dtypes(
        [None, None, np.uint32])
    _knl(queue, (globalSize,),
         (QuasarCL.maxWorkGroupSize,), out_g, sizes, size)
    cl.enqueue_copy(queue, output, out_g)
    return output




def calcCw(QuasarCL,wavelengthsMatrixFiltered, cReglinResults):
    queue = cl.CommandQueue(QuasarCL.ctx)
    
    h = wavelengthsMatrixFiltered.shape[0]  # spectrumsSize
    print(h)
    w = wavelengthsMatrixFiltered.shape[1]  # spectrumsNumber
    globalSize = QuasarCL.calcGlobalSize(h)

    continuum = np.zeros((h, w), dtype=np.double)

    wav_g = QuasarCL.readBuffer(wavelengthsMatrixFiltered)
    creg_g = QuasarCL.readBuffer(cReglinResults)

    cont_g = QuasarCL.writeBuffer(continuum)

    _knl = QuasarCL.buildKernel('continuum_kernels.cl').calc_cw
    _knl.set_scalar_arg_dtypes(
        [None, None, np.uint32, None])
    _knl(queue, (w, globalSize),
         (1, QuasarCL.maxWorkGroupSize), wav_g, cont_g, h, creg_g)
    cl.enqueue_copy(queue, continuum, cont_g)
    return continuum


def calcCfunDcfun(QuasarCL,
        wavelengthsMatrix,
        cReglinResultsVector,
        reglinResultsVector):
    queue = cl.CommandQueue(QuasarCL.ctx)
        
    h = wavelengthsMatrix.shape[0]  # spectrumsSize
    w = wavelengthsMatrix.shape[1]  # spectrumsNumber
    
    cReglinResultsVector = cReglinResultsVector.astype('double')
    reglinResultsVector = reglinResultsVector.astype('double')
    dContinuums = np.zeros((h, w), dtype=np.double)
    continuums = np.zeros((h, w), dtype=np.double)

    wav_g = QuasarCL.readBuffer(wavelengthsMatrix)
    dCon_g = QuasarCL.writeBuffer(dContinuums)
    con_g = QuasarCL.writeBuffer(continuums)
    cReg_g = QuasarCL.readBuffer(cReglinResultsVector)
    reg_g = QuasarCL.readBuffer(reglinResultsVector)

    _knl = QuasarCL.buildKernel('continuum_kernels.cl').calc_cfun_dcfun
    _knl(queue, (w, ASTRO_OBJ_SPEC_SIZE),
         (1, QuasarCL.maxWorkGroupSize), wav_g, dCon_g, con_g, cReg_g, reg_g)
    cl.enqueue_copy(queue, dContinuums, dCon_g)
    cl.enqueue_copy(queue, continuums, dCon_g)
    cont_dict= {"dContinuum":dContinuums,
                "continuum":continuums
               }
    return cont_dict


def outputMatrixTran(QuasarCL,w,h):
    outMat=np.zeros((w,h), dtype=np.float64)
    outMat = np.transpose(np.asarray(outMat, dtype=np.float64, order='F'))   
    out_g = QuasarCL.writeBuffer(outMat)
    return outMat,out_g

def outputMatrix(QuasarCL,w,h):
    outMat=np.zeros((w,h), dtype=np.float64)
    out_g = QuasarCL.writeBuffer(outMat)
    return outMat,out_g

#Merge filterWithWavelengthWindows,filterNonpositive,countIfNotInf,copyIfNotInf kernel functions 
def filter_matrix(QuasarCL,
        spectrumsMatrix,
        wavelengthsMatrix,
        errorsMatrix,
        sizes,
        windows,h,w,winSize):


    queue = cl.CommandQueue(QuasarCL.ctx)    
    
    _knl = QuasarCL.buildKernel('spectrums_kernels.cl').filterWithWavelengthWindows
    _knl.set_scalar_arg_dtypes(
        [None, None, None, None, None, np.uint32])
    _knl(queue, (w, ASTRO_OBJ_SPEC_SIZE), (1, QuasarCL.maxWorkGroupSize),
         wavelengthsMatrix, spectrumsMatrix, errorsMatrix, sizes, windows, winSize)
    

    sp = spectrumsMatrix.int_ptr
    wv = wavelengthsMatrix.int_ptr
    er = errorsMatrix.int_ptr
    
        
    _knl = QuasarCL.buildKernel('spectrums_kernels.cl').filterNonpositive
    _knl(queue, (w, ASTRO_OBJ_SPEC_SIZE),
         (1, QuasarCL.maxWorkGroupSize),cl.Buffer.from_int_ptr(sp),cl.Buffer.from_int_ptr(wv), cl.Buffer.from_int_ptr(er), sizes)
    
       
    ###
    newSizes = np.zeros(w, dtype=np.int32)
    sizes_g = QuasarCL.writeBuffer(newSizes)
    _knl = QuasarCL.buildKernel('tools_kernels.cl').countIfNotInf
    _knl._wg_info_cache = {}
    workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QuasarCL.devices[0])
    globalsize = calcGlobalSize(QuasarCL.maxWorkGroupSize, w)
    _knl.set_scalar_arg_dtypes([None, np.uint32, np.uint32, None])
    _knl(queue, (globalsize,),
         (workGroupMultiple,), cl.Buffer.from_int_ptr(sp), w, h, sizes_g)
    cl.enqueue_copy(queue, newSizes, sizes_g)
    
    maxs = max(newSizes)
    maxs = maxs if maxs > 0 else 64
    maxSize = QuasarCL.calcGlobalSize(maxs)
    
    ###
        
    outMat,out_g = outputMatrixTran(QuasarCL,w,maxSize)

    _knl = QuasarCL.buildKernel('tools_kernels.cl').copyIfNotInf
    _knl._wg_info_cache = {}
    workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QuasarCL.devices[0])
    globalsize = QuasarCL.calcGlobalSize(w)
    _knl.set_scalar_arg_dtypes([None, np.uint32, np.uint32, None, np.uint32])
    
    _knl(queue, (globalsize,),
         (workGroupMultiple,), cl.Buffer.from_int_ptr(sp), w, h, out_g, maxSize)
    cl.enqueue_copy(queue, outMat, out_g)
    specMat = outMat
    
    
    outMat,out_g = outputMatrixTran(QuasarCL,w,maxSize)
    _knl(queue, (globalsize,),
         (workGroupMultiple,), cl.Buffer.from_int_ptr(wv), w, h, out_g, maxSize)
    cl.enqueue_copy(queue, outMat, out_g)
    wavMat = outMat
        

    outMat,out_g = outputMatrixTran(QuasarCL,w,maxSize)
    _knl(queue, (globalsize,),
         (workGroupMultiple,), cl.Buffer.from_int_ptr(er), w, h, out_g, maxSize)
    cl.enqueue_copy(queue, outMat, out_g)
    errMat = outMat    
    
    out = {
    "wavelengthsMatrix":wavMat,
    "spectrumsMatrix":specMat,
    "errorsMatrix":errMat,
    "newSizes":newSizes,
    "maxSize":maxSize,
           }
    #print("continuum results:",out)
    return out


#Perform linear regression    
def reglin_results(QuasarCL,
        waveFiltered,
        specFiltered,ampWavelength,sizes,h,w):
    
    queue = cl.CommandQueue(QuasarCL.ctx)
    wav = waveFiltered.int_ptr
    cReglinResults,out_g = outputMatrix(QuasarCL,w,8)
     
    globalSize = QuasarCL.calcGlobalSize(w)
    _knl = QuasarCL.buildKernel('tools_kernels.cl').reglin
    _knl._wg_info_cache = {}
    workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QuasarCL.devices[0])
    _knl.set_scalar_arg_dtypes(
        [None, None, np.uint32, np.uint32, None, None])
    _knl(queue, (globalSize,),
         (workGroupMultiple,), waveFiltered, specFiltered, w, h, sizes, out_g)
    cr_g = out_g
    cl.enqueue_copy(queue, cReglinResults, out_g)
    size = len(cReglinResults)
     
    
    if ampWavelength > (math.pow(2.225074e-308, 10.001)):
        lampLog10 = math.log10(ampWavelength)
        _knl = QuasarCL.buildKernel('basics_kernels.cl').matrix_minus_scalar
        _knl.set_scalar_arg_dtypes(
                         [None, np.uint32, np.double])
        _knl(queue, (h, globalSize),
        (1, QuasarCL.maxWorkGroupSize), waveFiltered, w, lampLog10)
        wav = waveFiltered.int_ptr
        
    
    reglinResults,out_g = outputMatrix(QuasarCL,w,8)
    _knl = QuasarCL.buildKernel('tools_kernels.cl').reglin
    _knl._wg_info_cache = {}
    workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QuasarCL.devices[0])
    _knl.set_scalar_arg_dtypes(
        [None, None, np.uint32, np.uint32, None, None])
    _knl(queue, (globalSize,),
         (workGroupMultiple,), cl.Buffer.from_int_ptr(wav), specFiltered, w, h, sizes, out_g)
   
    r_g = out_g

    
    globalSize = QuasarCL.calcGlobalSize(size)
    cReg = np.zeros((size,8), dtype=np.double)
    reg = np.zeros((size,8), dtype=np.double)

    _knl = QuasarCL.buildKernel('continuum_kernels.cl').fix_reglin_results
    _knl.set_scalar_arg_dtypes(
        [None, None, np.uint32])
    _knl(queue, (globalSize,),
         (QuasarCL.maxWorkGroupSize,), cr_g, r_g, size)
    cl.enqueue_copy(queue, cReg,cr_g)
    cl.enqueue_copy(queue, reg, r_g)
    
    reglin_dict = {"cReglinResults":cReg,"reglinResults":reg}
    
    return reglin_dict          
 


    
