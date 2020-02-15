#!/usr/bin/env python

import pyopencl as cl
import numpy as np
ASTRO_OBJ_SPEC_SIZE = 4096


#To calculate the memory to be assigned for buffer
def calcGlobalSize(workGroupMultiple, dataSize):
    size = dataSize
    remainder = size % workGroupMultiple
    if (remainder != 0):
        size += workGroupMultiple - remainder
    if (size < dataSize):
        print("Error in calculating global_work_size.")
    return size

#To calculate base 10 log value of input matrix
def log10(QuasarCL,inpMat):
    queue = cl.CommandQueue(QuasarCL.ctx)
    
    h = inpMat.shape[0]  # spectrumsSize
    w = inpMat.shape[1]  # spectrumsNumber
    globalSize = QuasarCL.calcGlobalSize(w)

    outMat = np.zeros((w, h), dtype=np.float64)
    outMat = np.transpose(np.asarray(outMat, dtype=np.float64, order='F'))
    inp_g = QuasarCL.readBuffer(inpMat)

    _knl = QuasarCL.buildKernel('basics_kernels.cl').matrix_log10
    _knl(queue, (h, globalSize),
         (1, QuasarCL.maxWorkGroupSize), inp_g, np.uint32(w))
    cl.enqueue_copy(queue, outMat, inp_g)
    return outMat


#To subtract a scalar value from the input matrix
def minusScalar(QuasarCL,inpMat, subtrahend):
    queue = cl.CommandQueue(QuasarCL.ctx)
    
    h = inpMat.shape[0]  # spectrumsSize
    w = inpMat.shape[1]  # spectrumsNumber
    globalSize = QuasarCL.calcGlobalSize(w)

    inp_g = QuasarCL.readBuffer(inpMat)

    _knl = QuasarCL.buildKernel('basics_kernels.cl').matrix_minus_scalar
    _knl.set_scalar_arg_dtypes(
        [None, np.uint32, np.double])
    _knl(queue, (h, globalSize),
         (1, QuasarCL.maxWorkGroupSize), inp_g, w, subtrahend)
    cl.enqueue_copy(queue, inpMat, inp_g)
    return inpMat

#To subtract two matrices
def minusMatrix(QuasarCL,inputMatrix_,subtrahendMatrix_):
 queue = cl.CommandQueue(QuasarCL.ctx)
 
 h = inputMatrix_.shape[0]  # spectrumsSize
 w = inputMatrix_.shape[1]  # spectrumsNumber
 globalSize = QuasarCL.calcGlobalSize(w)
 outMat = np.zeros((w, h), dtype=np.float64)
 outMat = np.transpose(np.asarray(outMat, dtype=np.float64, order='F'))

 inp_g = QuasarCL.readBuffer(inputMatrix_)
 sub_g = QuasarCL.readBuffer(subtrahendMatrix_)
 output_g = QuasarCL.writeBuffer(outMat)

 _knl = QuasarCL.buildKernel('basics_kernels.cl').matrix_minus_matrix
 _knl(queue, (h, globalSize),
         (1, QuasarCL.maxWorkGroupSize), inp_g, np.uint32(w),sub_g,output_g)
 cl.enqueue_copy(queue, outMat, output_g)
 return outMat


#To multiply matrix with a vector
def multiplyCol(QuasarCL,inputMatrix_, vector_):
 queue = cl.CommandQueue(QuasarCL.ctx)
 
 h = inputMatrix_.shape[0]  # spectrumsSize
 w = inputMatrix_.shape[1]  # spectrumsNumber
 globalSize = QuasarCL.calcGlobalSize(w)
 outMat = np.zeros((w, h), dtype=np.float64)
 outMat = np.transpose(np.asarray(outMat, dtype=np.float64, order='F'))

 inp_g = QuasarCL.readBuffer(inputMatrix_)
 vec_g = QuasarCL.readBuffer(vector_)
 out_g = QuasarCL.writeBuffer(outMat)

 _knl = QuasarCL.buildKernel('basics_kernels.cl').matrix_multiply_col_vector
 _knl(queue, (h, globalSize),
         (1, QuasarCL.maxWorkGroupSize), inp_g, np.uint32(w),vec_g,out_g)
 cl.enqueue_copy(queue, outMat, out_g)
 return outMat

#To divide matrix by a matrix
def divideR(QuasarCL,inputMatrix_,divisorMatrix_):
 queue = cl.CommandQueue(QuasarCL.ctx)
 
 h = inputMatrix_.shape[0]  # spectrumsSize
 w = inputMatrix_.shape[1]  # spectrumsNumber
 globalSize = QuasarCL.calcGlobalSize(w)
 outMat = np.zeros((w, h), dtype=np.float64)
 outMat = np.transpose(np.asarray(outMat, dtype=np.float64, order='F'))

 inp_g = QuasarCL.readBuffer(inputMatrix_)
 div_g = QuasarCL.readBuffer(divisorMatrix_)
 out_g = QuasarCL.writeBuffer(outMat)

 _knl = QuasarCL.buildKernel('basics_kernels.cl').matrix_divide_matrix

 _knl(queue, (h, globalSize),
         (1, QuasarCL.maxWorkGroupSize), inp_g, np.uint32(w),div_g,out_g)
 cl.enqueue_copy(queue, outMat, out_g)
 return outMat

#To find transpose of a matrix
def matrix_transpose(QuasarCL,inpMat):
    queue = cl.CommandQueue(QuasarCL.ctx)
    
    h = inpMat.shape[0]  # spectrumsSize
    w = inpMat.shape[1]  # spectrumsNumber
    res_np = np.zeros((w, h), dtype=np.float64)
    res_np = np.transpose(np.asarray(res_np, dtype=np.float64, order='F'))
    BLOCK_DIM = 16
    globalSize1 = calcGlobalSize(BLOCK_DIM, h)
    globalSize2 = calcGlobalSize(BLOCK_DIM, w)

    a_g = QuasarCL.readBuffer(inpMat)
    res_g = QuasarCL.writeBuffer(res_np)

    _knl = QuasarCL.buildKernel('basics_kernels.cl').matrix_transpose3
    _knl.set_scalar_arg_dtypes([None, None, np.uint32, np.uint32])
    _knl(queue, (w, h), None, a_g, res_g, np.uint32(w), np.uint32(h))
    cl.enqueue_copy(queue, res_np, res_g)
    return res_np