#!/usr/bin/env python

import pyopencl as cl
import numpy as np
ASTRO_OBJ_SPEC_SIZE = 4096

#To filter infinity values from spectrum

def filterInfs(QuasarCL,spectrumsMatrix_, aMatrix_, bMatrix_, sizesVector_):
 queue = cl.CommandQueue(QuasarCL.ctx)
 
 h = spectrumsMatrix_.shape[0]  # spectrumsSize
 w = spectrumsMatrix_.shape[1]  # spectrumsNumber
 spec = spectrumsMatrix_
 a = aMatrix_
 b = bMatrix_

 spec_g = QuasarCL.makeBuffer(spectrumsMatrix_)
 a_g =QuasarCL.makeBuffer(aMatrix_)
 b_g = QuasarCL.makeBuffer(bMatrix_)
 sizes_g = QuasarCL.makeBuffer(sizesVector_)
 
 _knl = QuasarCL.buildKernel('spectrums_kernels.cl').filterInfs
 
 _knl(queue, (w,ASTRO_OBJ_SPEC_SIZE), (1,QuasarCL.maxWorkGroupSize), spec_g, a_g, b_g, sizes_g)
 cl.enqueue_copy(queue, spec, spec_g)
 cl.enqueue_copy(queue, a, a_g)
 cl.enqueue_copy(queue, b, b_g)
 out = {
    "spectrumsMatrix":spec,
    "aMatrix":a,
    "bMatrix":b
     }
 return out

#To filter zeros from the input matrices

def filterZeros(QuasarCL,inpMat, MatA, MatB, sizes):
    queue = cl.CommandQueue(QuasarCL.ctx)
    
    h = inpMat.shape[0]  # spectrumsSize
    w = inpMat.shape[1]  # spectrumsNumber

    spec = np.zeros((w, h), dtype=np.float64)
    spec = np.transpose(np.asarray(spec, dtype=np.float64, order='F'))
    a = np.zeros((w, h), dtype=np.float64)
    a = np.transpose(np.asarray(a, dtype=np.float64, order='F'))
    b = np.zeros((w, h), dtype=np.float64)
    b = np.transpose(np.asarray(b, dtype=np.float64, order='F'))

    inp_g = QuasarCL.readBuffer(inpMat)
    a_g = QuasarCL.readBuffer(MatA)
    b_g = QuasarCL.readBuffer(MatB)
    bufferSizes = QuasarCL.readBuffer(sizes)

    _knl = QuasarCL.buildKernel('spectrums_kernels.cl').filterZeros

    _knl(queue, (w, ASTRO_OBJ_SPEC_SIZE),
         (1, QuasarCL.maxWorkGroupSize), inp_g, a_g, b_g, bufferSizes)
    cl.enqueue_copy(queue, spec, inp_g)
    cl.enqueue_copy(queue, a, a_g)
    cl.enqueue_copy(queue, b, b_g)
    out = {
    "spectrumsMatrix":spec,
    "aMatrix":a,
    "bMatrix":b
     }
    return out

#To filter the spectrum using a window
def filterWithWavelengthWindows(QuasarCL,
        spectrumsMatrix,
        wavelengthsMatrix,
        errorsMatrix,
        sizes,
        windows):
    queue = cl.CommandQueue(QuasarCL.ctx)
    
    
    h = spectrumsMatrix.shape[0]  # spectrumsSize
    w = spectrumsMatrix.shape[1]  # spectrumsNumber
    
    windowsVector = np.asarray(windows)
    winSize = len(windowsVector)

    wavMat = np.zeros((w, h), dtype=np.float64)
    wavMat = np.transpose(np.asarray(wavMat, dtype=np.float64, order='F'))
    specMat = np.zeros((w, h), dtype=np.float64)
    specMat = np.transpose(np.asarray(specMat, dtype=np.float64, order='F'))
    errMat = np.zeros((w, h), dtype=np.float64)
    errMat = np.transpose(np.asarray(errMat, dtype=np.float64, order='F'))

    sMat = QuasarCL.readBuffer(spectrumsMatrix)
    wMat = QuasarCL.readBuffer(wavelengthsMatrix)
    eMat = QuasarCL.readBuffer(errorsMatrix)
    sizVec = QuasarCL.readBuffer(sizes)
    winVec = QuasarCL.readBuffer(windowsVector)

    _knl = QuasarCL.buildKernel('spectrums_kernels.cl').filterWithWavelengthWindows

    _knl.set_scalar_arg_dtypes(
        [None, None, None, None, None, np.uint32])
    _knl(queue, (w, ASTRO_OBJ_SPEC_SIZE), (1, QuasarCL.maxWorkGroupSize),
         wMat, sMat, eMat, sizVec, winVec, winSize)
    cl.enqueue_copy(queue, wavMat, wMat)
    cl.enqueue_copy(queue, specMat, sMat)
    cl.enqueue_copy(queue, errMat, eMat)
    out = {
    "wavelengthsMatrix":wavMat,
    "spectrumsMatrix":specMat,
    "errorsMatrix":errMat
     }
    return out


#To filter negative values
def filterNonpositive(QuasarCL,specMat, wavMat, errMat, sizesVector):
    queue = cl.CommandQueue(QuasarCL.ctx)
    
    
    h = specMat.shape[0]  # spectrumsSize
    w = specMat.shape[1]  # spectrumsNumber

    wMat = np.zeros((w, h), dtype=np.float64)
    wMat = np.transpose(np.asarray(wMat, dtype=np.float64, order='F'))
    sMat = np.zeros((w, h), dtype=np.float64)
    sMat = np.transpose(np.asarray(sMat, dtype=np.float64, order='F'))
    eMat = np.zeros((w, h), dtype=np.float64)
    eMat = np.transpose(np.asarray(eMat, dtype=np.float64, order='F'))

    s_g = QuasarCL.readBuffer(specMat)
    w_g = QuasarCL.readBuffer(wavMat)
    e_g = QuasarCL.readBuffer(errMat)
    sizVec = QuasarCL.readBuffer(sizesVector)

    _knl = QuasarCL.buildKernel('spectrums_kernels.cl').filterNonpositive

    _knl(queue, (w, ASTRO_OBJ_SPEC_SIZE),
         (1, QuasarCL.maxWorkGroupSize), s_g, w_g, e_g, sizVec)
    cl.enqueue_copy(queue, wMat, w_g)
    cl.enqueue_copy(queue, sMat, s_g)
    cl.enqueue_copy(queue, eMat, e_g)
    out = {
    "wavelengthsMatrix":wMat,
    "spectrumsMatrix":sMat,
    "errorsMatrix":eMat
     }
    return out


#To aggregate the spectrums
def addSpectrum(QuasarCL,spectrumsMatrix,wavelengthsMatrix,sizes,size,toAddWavelengths,
                toAddSpectrum,toAddSize):
 queue = cl.CommandQueue(QuasarCL.ctx)
 
 output = spectrumsMatrix
 globalSize = QuasarCL.calcGlobalSize(size)
 wav_g = QuasarCL.readBuffer(wavelengthsMatrix)
 spec_g = QuasarCL.readBuffer(spectrumsMatrix)
 sizes_g = QuasarCL.readBuffer(sizes)
 toAddwav_g = QuasarCL.readBuffer(toAddWavelengths)
 toAddsp_g = QuasarCL.readBuffer(toAddSpectrum)
 output_g = QuasarCL.writeBuffer(output)

 _knl = QuasarCL.buildKernel('spectrums_kernels.cl').addSpectrum
 _knl.set_scalar_arg_dtypes(
        [None, None, None, np.uint32, None, None,np.uint32,None])
 _knl(queue, (globalSize,),
         (QuasarCL.maxWorkGroupSize,), wav_g, spec_g, sizes_g, size,toAddwav_g,toAddsp_g ,toAddSize,output_g)
 cl.enqueue_copy(queue, output, output_g)
 return output





