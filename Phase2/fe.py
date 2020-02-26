#!/usr/bin/env python

import pyopencl as cl
import numpy as np
from spectrum import *
from tools import *
from basics import *
ASTRO_OBJ_SPEC_SIZE = 4096


def calcReducedChisqs(QsoCL,fMatrix_,yMatrix_,errorsMatrix_,sizesVector_):
 queue = cl.CommandQueue(QsoCL.ctx)

 h = fMatrix_.shape[0]  # spectrumsSize
 w = fMatrix_.shape[1]  # spectrumsNumber
 globalSize = QsoCL.calcGlobalSize(w)
 chisqResults= chisqR(QsoCL,fMatrix_,yMatrix_,errorsMatrix_,sizesVector_)
 print("after chisqR interfun")
 print(chisqResults)
 print(sizesVector_)
 chisq_g = QsoCL.makeBuffer(chisqResults)
 sizes_g = QsoCL.readBuffer(sizesVector_)

 _knl = QsoCL.buildKernel('fe_kernels.cl').reduce_fe_chisqs
 _knl.set_scalar_arg_dtypes(
        [None, None,np.uint32])
 _knl(queue, (globalSize,),
         (QsoCL.maxWorkGroupSize,), chisq_g, sizes_g, w)
 cl.enqueue_copy(queue, chisqResults, chisq_g)
 return chisqResults



def copyIfNotInf_(QsoCL,inp_g, filteredsize,h,w):
    queue = cl.CommandQueue(QsoCL.ctx)
    
    outMat=np.zeros((w,h), dtype=np.float64)
    outMat = np.transpose(np.asarray(outMat, dtype=np.float64, order='F'))   
    res_g = QsoCL.writeBuffer(outMat)
    globalsize = QsoCL.calcGlobalSize(w)

    _knl = QsoCL.buildKernel('tools_kernels.cl').copyIfNotInf

    _knl._wg_info_cache = {}
    workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
    _knl.set_scalar_arg_dtypes([None, np.uint32, np.uint32, None, np.uint32])
    _knl(queue, (globalsize,),
         (workGroupMultiple,), inp_g, w, h, res_g, filteredsize)
    
    return res_g




def outputMatrixTran(QsoCL,w,h):
    outMat=np.zeros((w,h), dtype=np.float64)
    outMat = np.transpose(np.asarray(outMat, dtype=np.float64, order='F'))   
    out_g = QsoCL.writeBuffer(outMat)
    return outMat,out_g


#To filter matrices with infinity values and perform linear regression followed by chi-squared test 
def filtered_chisqs(QsoCL,wavelengthsMatrix,errorsMatrix,spectrumsMatrix, continuumsMatrix, templateFeMatrix, sizes,fitParameters,h,w):
 
 #filteredMatrices = filterInfs(QsoCL,spectrumsMatrix, continuumsMatrix, templateFeMatrix, sizes)
 queue = cl.CommandQueue(QsoCL.ctx)
 globalSize = QsoCL.calcGlobalSize(w)
 
 _knl = QsoCL.buildKernel('spectrums_kernels.cl').filterInfs
 _knl(queue, (w,ASTRO_OBJ_SPEC_SIZE), (1,QsoCL.maxWorkGroupSize), spectrumsMatrix, continuumsMatrix, templateFeMatrix, sizes)
 
 spec = spectrumsMatrix.int_ptr
 cont = continuumsMatrix.int_ptr
 tempFe = templateFeMatrix.int_ptr
 wav = wavelengthsMatrix.int_ptr
 err = errorsMatrix.int_ptr
 
  
 if(fitParameters['fitType']=='FWIN'):
  _knl = QsoCL.buildKernel('spectrums_kernels.cl').filterZeros
  _knl(queue, (w, ASTRO_OBJ_SPEC_SIZE),
         (1, QsoCL.maxWorkGroupSize), tempFe, wav, err, sizes)
  
  _knl = QsoCL.buildKernel('spectrums_kernels.cl').filterInfs 
  _knl(queue, (w,ASTRO_OBJ_SPEC_SIZE), (1,QsoCL.maxWorkGroupSize), tempFe, cont, spec, sizes)
  
 
 spectrumsMatrixFiltered = cl.Buffer.from_int_ptr(spec)
 wavelengthsMatrixFiltered = cl.Buffer.from_int_ptr(wav)
 errorsMatrixFiltered = cl.Buffer.from_int_ptr(err)
 continuumsMatrixFiltered = cl.Buffer.from_int_ptr(cont)
 templateFeMatrixFiltered = cl.Buffer.from_int_ptr(tempFe)
 sizesFeWindows = sizes
 
 tempFeMatrix,out_g = outputMatrixTran(QsoCL,w,h)
 cl.enqueue_copy(queue, tempFeMatrix, templateFeMatrixFiltered)
 
 if(fitParameters['fitType']=='WIN'or fitParameters['fitType']=='FWIN'):  
  sizesFeWindows = countIfNotInf(QsoCL,tempFeMatrix)
  maxSize = max(sizesFeWindows)
  spectrumsMatrixFiltered = copyIfNotInf_(QsoCL,spectrumsMatrix, maxSize,h,w)
  wavelengthsMatrixFiltered = copyIfNotInf_(QsoCL,wavelengthsMatrix, maxSize,h,w)
  errorsMatrixFiltered = copyIfNotInf_(QsoCL,errorsMatrix, maxSize,h,w)
  continuumsMatrixFiltered = copyIfNotInf_(QsoCL,continuumsMatrix, maxSize,h,w)
  templateFeMatrixFiltered = copyIfNotInf_(QsoCL,templateFeMatrix, maxSize,h,w) 
 
 print("before scale")
  
 reglinYaxResultsVec = np.transpose(np.zeros(w,dtype=np.float64,order='F'))
 regYax = QsoCL.makeBuffer(reglinYaxResultsVec)
 
 _knl = QsoCL.buildKernel('tools_kernels.cl').reglin_yax
 _knl._wg_info_cache = {}
 workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
 _knl.set_scalar_arg_dtypes(
        [None, None,np.uint32, np.uint32, None,None])
 _knl(queue, (globalSize,),
         (workGroupMultiple,), templateFeMatrixFiltered, spectrumsMatrixFiltered, w, h,sizes,regYax)
 cl.enqueue_copy(queue, reglinYaxResultsVec, regYax)
 
 for i in range(len(reglinYaxResultsVec)):
   reglinYaxResultsVec[i] = reglinYaxResultsVec[i]*fitParameters['feScaleRate']
 
 
 scaleRates = reglinYaxResultsVec
 reglinVec = QsoCL.readBuffer(reglinYaxResultsVec)
  
 outMat,fe_g = outputMatrixTran(QsoCL,w,h)
 _knl = QsoCL.buildKernel('basics_kernels.cl').matrix_multiply_col_vector
 _knl(queue, (h, globalSize),
         (1, QsoCL.maxWorkGroupSize), templateFeMatrixFiltered, np.uint32(w),reglinVec,fe_g)

 templateFeMatrixCopy,out_g = outputMatrixTran(QsoCL,w,h)
 _knl = QsoCL.buildKernel('basics_kernels.cl').matrix_multiply_col_vector
 _knl(queue, (h, globalSize),
         (1, QsoCL.maxWorkGroupSize), cl.Buffer.from_int_ptr(tempFe), np.uint32(w),reglinVec,out_g)
 
 cl.enqueue_copy(queue, templateFeMatrixCopy, out_g) 
 

 #chisqs
 chisqResults,chisq_g = outputMatrixTran(QsoCL,w,h)
 _knl = QsoCL.buildKernel('tools_kernels.cl').chisq
 _knl._wg_info_cache = {}
 workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
 _knl.set_scalar_arg_dtypes(
        [None, None, None, np.uint32, np.uint32, None, None])
 _knl(queue, (globalSize,),
         (workGroupMultiple,), spectrumsMatrixFiltered, fe_g, errorsMatrixFiltered, w, h, sizes, chisq_g)

 _knl = QsoCL.buildKernel('fe_kernels.cl').reduce_fe_chisqs
 _knl.set_scalar_arg_dtypes(
        [None, None,np.uint32])
 _knl(queue, (globalSize,),
         (QsoCL.maxWorkGroupSize,), chisq_g, sizes, w)
 cl.enqueue_copy(queue, chisqResults, chisq_g)
 
 return templateFeMatrixCopy,chisqResults,scaleRates,sizesFeWindows
 
