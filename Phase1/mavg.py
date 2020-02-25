#!/usr/bin/env python

import pyopencl as cl
import numpy as np
ASTRO_OBJ_SPEC_SIZE = 4096


#To calculate simple moving average
def simple_mavg(QsoCL,inpMat, window):
    queue = cl.CommandQueue(QsoCL.ctx)
    
    h = inpMat.shape[0]  # spectrumsSize
    w = inpMat.shape[1]  # spectrumsNumber
    globalSize = QsoCL.calcGlobalSize(w)
    res_np = np.zeros((h, w), dtype=np.float64)
    a_g =  QsoCL.readBuffer(inpMat)
    res_g = QsoCL.writeBuffer(res_np)

    _knl = QsoCL.buildKernel('mavg_kernels.cl').simple_mavg

    _knl.set_scalar_arg_dtypes([None, np.uint32, np.uint32, None, np.uint32])
    _knl(queue, (globalSize,), (QsoCL.maxWorkGroupSize,),
         a_g, w, h, res_g, window)
    cl.enqueue_copy(queue, res_np, res_g)
    #res_np = np.transpose(res_np)
    return res_np


#To calculate centered moving average

def centered_mavg(QsoCL,inpMat, sizes, window):
    queue = cl.CommandQueue(QsoCL.ctx)
    
    h = inpMat.shape[0]  # spectrumsSize
    w = inpMat.shape[1]  # spectrumsNumber
    globalSize = QsoCL.calcGlobalSize(w)
    res_np = np.zeros((h, w), dtype=np.float64)

    a_g = QsoCL.readBuffer(inpMat)
    bufferSizes = QsoCL.readBuffer(sizes)
    res_g = QsoCL.writeBuffer(res_np)

    _knl = QsoCL.buildKernel('mavg_kernels.cl').centered_mavg
    _knl.set_scalar_arg_dtypes(
        [None, np.uint32, np.uint32, None, None, np.uint32])
    _knl(queue, (globalSize,), (QsoCL.maxWorkGroupSize,),
         a_g, w, h, bufferSizes, res_g, window)
    cl.enqueue_copy(queue, res_np, res_g)
    return res_np
