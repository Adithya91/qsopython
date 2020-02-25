import pyopencl as cl
import numpy as np
from QsoCL import *
ASTRO_OBJ_SPEC_SIZE = 4096


def calcGlobalSize(workGroupMultiple, dataSize):
    size = dataSize
    remainder = size % workGroupMultiple
    if (remainder != 0):
        size += workGroupMultiple - remainder
    if (size < dataSize):
        print("Error in calculating global_work_size.")
    return size

#fix the linear regression results
def fixReglinResults(QsoCL,cReglinResults, reglinResults):
    queue = cl.CommandQueue(QsoCL.ctx)
    
    size = len(cReglinResults)
    print(size)
    cReg = np.zeros((size,8), dtype=np.double)
    reg =  np.zeros((size,8), dtype=np.double)
    cr_g = QsoCL.makeBuffer(cReglinResults)
    r_g = QsoCL.makeBuffer(reglinResults)
    globalSize = QsoCL.calcGlobalSize(size)

    _knl = QsoCL.buildKernel('continuum_kernels.cl').fix_reglin_results
    _knl.set_scalar_arg_dtypes(
        [None, None, np.uint32])
    _knl(queue, (globalSize,),
         (QsoCL.maxWorkGroupSize,), cr_g, r_g, size)
    cl.enqueue_copy(queue, cReg, cr_g)
    cl.enqueue_copy(queue, reg, r_g)
    reglin_dict = {"cReglinResults":cReg,"reglinResults":reg}
    return reglin_dict


def calcCw(QsoCL,wavelengthsMatrixFiltered, cReglinResults):
    queue = cl.CommandQueue(QsoCL.ctx)
    
    h = wavelengthsMatrixFiltered.shape[0]  # spectrumsSize
    print(h)
    w = wavelengthsMatrixFiltered.shape[1]  # spectrumsNumber
    globalSize = QsoCL.calcGlobalSize(h)

    continuum = np.zeros((h, w), dtype=np.double)

    wav_g = QsoCL.readBuffer(wavelengthsMatrixFiltered)
    creg_g = QsoCL.readBuffer(cReglinResults)

    cont_g = QsoCL.writeBuffer(continuum)

    _knl = QsoCL.buildKernel('continuum_kernels.cl').calc_cw
    _knl.set_scalar_arg_dtypes(
        [None, None, np.uint32, None])
    _knl(queue, (w, globalSize),
         (1, QsoCL.maxWorkGroupSize), wav_g, cont_g, h, creg_g)
    cl.enqueue_copy(queue, continuum, cont_g)
    return continuum

#filter the continuum with Chi-squared test
def reduceContinuumChisqs(QsoCL,chisqs, sizesVector):
    queue = cl.CommandQueue(QsoCL.ctx)
    
    size = len(chisqs)
    chis = chisqs
    chisqs_g = QsoCL.readBuffer(chis)
    sizes = QsoCL.readBuffer(sizesVector)
    
    globalSize = QsoCL.calcGlobalSize(size)

    _knl = QsoCL.buildKernel('continuum_kernels.cl').reduce_continuum_chisqs

    _knl.set_scalar_arg_dtypes(
        [None, None, np.uint32])
    _knl(queue, (globalSize,),
         (QsoCL.maxWorkGroupSize,), chisqs_g, sizes, size)
    cl.enqueue_copy(queue, chis, chisqs_g)
    return chis

def calcCfunDcfun(QsoCL,
        wavelengthsMatrix,
        cReglinResultsVector,
        reglinResultsVector):
    queue = cl.CommandQueue(QsoCL.ctx)
        
    h = wavelengthsMatrix.shape[0]  # spectrumsSize
    w = wavelengthsMatrix.shape[1]  # spectrumsNumber
    
    cReglinResultsVector = cReglinResultsVector.astype('double')
    reglinResultsVector = reglinResultsVector.astype('double')
    dContinuums = np.zeros((h, w), dtype=np.double)
    continuums = np.zeros((h, w), dtype=np.double)

    wav_g = QsoCL.readBuffer(wavelengthsMatrix)
    dCon_g = QsoCL.writeBuffer(dContinuums)
    con_g = QsoCL.writeBuffer(continuums)
    cReg_g = QsoCL.readBuffer(cReglinResultsVector)
    reg_g = QsoCL.readBuffer(reglinResultsVector)

    _knl = QsoCL.buildKernel('continuum_kernels.cl').calc_cfun_dcfun
    _knl(queue, (w, ASTRO_OBJ_SPEC_SIZE),
         (1, QsoCL.maxWorkGroupSize), wav_g, dCon_g, con_g, cReg_g, reg_g)
    cl.enqueue_copy(queue, dContinuums, dCon_g)
    cl.enqueue_copy(queue, continuums, dCon_g)
    cont_dict= {"dContinuum":dContinuums,
                "continuum":continuums
               }
    return cont_dict
