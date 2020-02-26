#!/usr/bin/env python

import pyopencl as cl
import numpy as np
from basics import *
ASTRO_OBJ_SPEC_SIZE = 4096

def copyIfNotInf(QsoCL,inpMat, filteredsize):
    queue = cl.CommandQueue(QsoCL.ctx)
    
    h = inpMat.shape[0]  # spectrumsSize
    w = inpMat.shape[1]  # spectrumsNumber

    outMat = np.zeros((w,filteredsize), dtype=np.float64)
    outMat = np.transpose(np.asarray(outMat, dtype=np.float64, order='F'))
    inp_g = QsoCL.readBuffer(inpMat)
    globalsize = QsoCL.calcGlobalSize(w)

    res_g = QsoCL.writeBuffer(outMat)

    _knl = QsoCL.buildKernel('tools_kernels.cl').copyIfNotInf

    _knl._wg_info_cache = {}
    workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
    _knl.set_scalar_arg_dtypes([None, np.uint32, np.uint32, None, np.uint32])
    _knl(queue, (globalsize,),
         (workGroupMultiple,), inp_g, w, h, res_g, filteredsize)
    cl.enqueue_copy(queue, outMat, res_g)
    return outMat

def countIfNotInf(QsoCL, inpMat):
    
    queue = cl.CommandQueue(QsoCL.ctx)
    
    h = inpMat.shape[0]  # spectrumsSize
    w = inpMat.shape[1]  # spectrumsNumber
    
    globalsize = calcGlobalSize(QsoCL.maxWorkGroupSize, w)
    sizes = np.zeros(w, dtype=np.int32)
    inp_g = QsoCL.readBuffer(inpMat)
    

    res_g = QsoCL.writeBuffer(sizes)

    _knl = QsoCL.buildKernel('tools_kernels.cl').countIfNotInf

    _knl._wg_info_cache = {}
    workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
    _knl.set_scalar_arg_dtypes([None, np.uint32, np.uint32, None])
    _knl(queue, (globalsize,),
         (workGroupMultiple,), inp_g, w, h, res_g)
    cl.enqueue_copy(queue, sizes, res_g)
    return sizes

#Trapezoidal numerical integration is approximating the integration over an 
#interval by breaking the area into trapezoids.
def trapz(QsoCL,yMatrix_, xMatrix_, sizesVector_):
 queue = cl.CommandQueue(QsoCL.ctx)

 h = yMatrix_.shape[0]  # spectrumsSize
 w = yMatrix_.shape[1]  # spectrumsNumber

 globalSize = QsoCL.calcGlobalSize(w)
 outMat = np.zeros((h,w), dtype=np.float64)

 y_g = QsoCL.readBuffer(yMatrix_)

 x_g = QsoCL.readBuffer(xMatrix_)

 sizes_g = QsoCL.readBuffer(sizesVector_)

 out_g = QsoCL.writeBuffer(outMat)

 _knl = QsoCL.buildKernel('tools_kernels.cl').integrate_trapz
 _knl._wg_info_cache = {}
 workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])

 _knl(queue, (globalSize,),
         (workGroupMultiple,), y_g, x_g,np.uint32(w),np.uint32(h),sizes_g,out_g)
 cl.enqueue_copy(queue, outMat, out_g)
 return outMat


#To perform linear regression
def reglinR(QsoCL,xMat, yMat, sizesVector):
    queue = cl.CommandQueue(QsoCL.ctx)
    
    h = xMat.shape[0]  # spectrumsSize
    w = xMat.shape[1]  # spectrumsNumber

    globalSize = QsoCL.calcGlobalSize(w)
    output = np.zeros((w,8), dtype=np.float64)

    x_g = QsoCL.readBuffer(xMat)
    y_g = QsoCL.readBuffer(yMat)
    sizes = QsoCL.readBuffer(sizesVector)

    res_g = QsoCL.writeBuffer(output)

    _knl = QsoCL.buildKernel('tools_kernels.cl').reglin
    _knl._wg_info_cache = {}
    workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
    _knl.set_scalar_arg_dtypes(
        [None, None, np.uint32, np.uint32, None, None])
    _knl(queue, (globalSize,),
         (workGroupMultiple,), x_g, y_g, w, h, sizes, res_g)
    cl.enqueue_copy(queue, output, res_g)

    return output

#Chi - Squared Test is a function used for testing the goodness of fit (x**2 test)
def chisqR(QsoCL,fMat, yMat, errorsMatrix, sizesVector):
    queue = cl.CommandQueue(QsoCL.ctx)
    
    h = fMat.shape[0]  # spectrumsSize
    w = fMat.shape[1]  # spectrumsNumber
    output = np.zeros(w, dtype=np.double)

    f_g = QsoCL.readBuffer(fMat)
    y_g = QsoCL.readBuffer(yMat)
    e_g = QsoCL.readBuffer(errorsMatrix)
    sizVec = QsoCL.readBuffer(sizesVector)

    globalSize = QsoCL.calcGlobalSize(w)
    res_g = QsoCL.writeBuffer(output)

    _knl = QsoCL.buildKernel('tools_kernels.cl').chisq
    _knl._wg_info_cache = {}
    workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
    _knl.set_scalar_arg_dtypes(
        [None, None, None, np.uint32, np.uint32, None, None])
    _knl(queue, (globalSize,),
         (workGroupMultiple,), f_g, y_g, e_g, w, h, sizVec, res_g)
    cl.enqueue_copy(queue, output, res_g)
    return output


def reglinYax(QsoCL,xs,ys,width,height,sizes,output):
 queue = cl.CommandQueue(QsoCL.ctx)
 
 globalSize = QsoCL.calcGlobalSize(width)
 out = output
 x_g = QsoCL.readBuffer(xs)
 y_g = QsoCL.readBuffer(ys)
 sizes_g = QsoCL.readBuffer(sizes)
 out_g = QsoCL.writeBuffer(out)
 
 _knl = QsoCL.buildKernel('tools_kernels.cl').reglin_yax
 _knl._wg_info_cache = {}
 workGroupMultiple = _knl.get_work_group_info(
        cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        QsoCL.devices[0])
 _knl.set_scalar_arg_dtypes(
        [None, None,np.uint32, np.uint32, None,None])
 _knl(queue, (globalSize,),
         (workGroupMultiple,), x_g, y_g, width, height,sizes_g,out_g)
 cl.enqueue_copy(queue, out, out_g)
 return out


