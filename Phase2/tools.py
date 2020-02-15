#!/usr/bin/env python

import pyopencl as cl
import numpy as np
from basics import *
ASTRO_OBJ_SPEC_SIZE = 4096

def copyIfNotInf(QuasarCL,inpMat, filteredsize):
    queue = cl.CommandQueue(QuasarCL.ctx)
    
    h = inpMat.shape[0]  # spectrumsSize
    w = inpMat.shape[1]  # spectrumsNumber

    outMat = np.zeros((w,filteredsize), dtype=np.float64)
    outMat = np.transpose(np.asarray(outMat, dtype=np.float64, order='F'))
    inp_g = QuasarCL.readBuffer(inpMat)
    globalsize = QuasarCL.calcGlobalSize(w)

    res_g = QuasarCL.writeBuffer(outMat)

    _knl = QuasarCL.buildKernel('tools_kernels.cl').copyIfNotInf

    _knl._wg_info_cache = {}
    workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QuasarCL.devices[0])
    _knl.set_scalar_arg_dtypes([None, np.uint32, np.uint32, None, np.uint32])
    _knl(queue, (globalsize,),
         (workGroupMultiple,), inp_g, w, h, res_g, filteredsize)
    cl.enqueue_copy(queue, outMat, res_g)
    return outMat

def countIfNotInf(QuasarCL, inpMat):
    
    queue = cl.CommandQueue(QuasarCL.ctx)
    
    h = inpMat.shape[0]  # spectrumsSize
    w = inpMat.shape[1]  # spectrumsNumber
    
    globalsize = calcGlobalSize(QuasarCL.maxWorkGroupSize, w)
    sizes = np.zeros(w, dtype=np.int32)
    inp_g = QuasarCL.readBuffer(inpMat)
    

    res_g = QuasarCL.writeBuffer(sizes)

    _knl = QuasarCL.buildKernel('tools_kernels.cl').countIfNotInf

    _knl._wg_info_cache = {}
    workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QuasarCL.devices[0])
    _knl.set_scalar_arg_dtypes([None, np.uint32, np.uint32, None])
    _knl(queue, (globalsize,),
         (workGroupMultiple,), inp_g, w, h, res_g)
    cl.enqueue_copy(queue, sizes, res_g)
    return sizes

#Trapezoidal numerical integration is approximating the integration over an 
#interval by breaking the area into trapezoids.
def trapz(QuasarCL,yMatrix_, xMatrix_, sizesVector_):
 queue = cl.CommandQueue(QuasarCL.ctx)

 h = yMatrix_.shape[0]  # spectrumsSize
 w = yMatrix_.shape[1]  # spectrumsNumber

 globalSize = QuasarCL.calcGlobalSize(w)
 outMat = np.zeros((h,w), dtype=np.float64)

 y_g = QuasarCL.readBuffer(yMatrix_)

 x_g = QuasarCL.readBuffer(xMatrix_)

 sizes_g = QuasarCL.readBuffer(sizesVector_)

 out_g = QuasarCL.writeBuffer(outMat)

 _knl = QuasarCL.buildKernel('tools_kernels.cl').integrate_trapz
 _knl._wg_info_cache = {}
 workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QuasarCL.devices[0])

 _knl(queue, (globalSize,),
         (workGroupMultiple,), y_g, x_g,np.uint32(w),np.uint32(h),sizes_g,out_g)
 cl.enqueue_copy(queue, outMat, out_g)
 return outMat


#To perform linear regression
def reglinR(QuasarCL,xMat, yMat, sizesVector):
    queue = cl.CommandQueue(QuasarCL.ctx)
    
    h = xMat.shape[0]  # spectrumsSize
    w = xMat.shape[1]  # spectrumsNumber

    globalSize = QuasarCL.calcGlobalSize(w)
    output = np.zeros((w,8), dtype=np.float64)

    x_g = QuasarCL.readBuffer(xMat)
    y_g = QuasarCL.readBuffer(yMat)
    sizes = QuasarCL.readBuffer(sizesVector)

    res_g = QuasarCL.writeBuffer(output)

    _knl = QuasarCL.buildKernel('tools_kernels.cl').reglin
    _knl._wg_info_cache = {}
    workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QuasarCL.devices[0])
    _knl.set_scalar_arg_dtypes(
        [None, None, np.uint32, np.uint32, None, None])
    _knl(queue, (globalSize,),
         (workGroupMultiple,), x_g, y_g, w, h, sizes, res_g)
    cl.enqueue_copy(queue, output, res_g)

    return output

#Chi - Squared Test is a function used for testing the goodness of fit (x**2 test)
def chisqR(QuasarCL,fMat, yMat, errorsMatrix, sizesVector):
    queue = cl.CommandQueue(QuasarCL.ctx)
    
    h = fMat.shape[0]  # spectrumsSize
    w = fMat.shape[1]  # spectrumsNumber
    output = np.zeros(w, dtype=np.double)

    f_g = QuasarCL.readBuffer(fMat)
    y_g = QuasarCL.readBuffer(yMat)
    e_g = QuasarCL.readBuffer(errorsMatrix)
    sizVec = QuasarCL.readBuffer(sizesVector)

    globalSize = QuasarCL.calcGlobalSize(w)
    res_g = QuasarCL.writeBuffer(output)

    _knl = QuasarCL.buildKernel('tools_kernels.cl').chisq
    _knl._wg_info_cache = {}
    workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QuasarCL.devices[0])
    _knl.set_scalar_arg_dtypes(
        [None, None, None, np.uint32, np.uint32, None, None])
    _knl(queue, (globalSize,),
         (workGroupMultiple,), f_g, y_g, e_g, w, h, sizVec, res_g)
    cl.enqueue_copy(queue, output, res_g)
    return output


def reglinYax(QuasarCL,xs,ys,width,height,sizes,output):
 queue = cl.CommandQueue(QuasarCL.ctx)
 
 globalSize = QuasarCL.calcGlobalSize(width)
 out = output
 x_g = QuasarCL.readBuffer(xs)
 y_g = QuasarCL.readBuffer(ys)
 sizes_g = QuasarCL.readBuffer(sizes)
 out_g = QuasarCL.writeBuffer(out)
 
 _knl = QuasarCL.buildKernel('tools_kernels.cl').reglin_yax
 _knl._wg_info_cache = {}
 workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QuasarCL.devices[0])
 _knl.set_scalar_arg_dtypes(
        [None, None,np.uint32, np.uint32, None,None])
 _knl(queue, (globalSize,),
         (workGroupMultiple,), x_g, y_g, width, height,sizes_g,out_g)
 cl.enqueue_copy(queue, out, out_g)
 return out


