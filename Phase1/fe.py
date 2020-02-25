import pyopencl as cl
import numpy as np
from tools import chisqR
ASTRO_OBJ_SPEC_SIZE = 4096

#To filter iron templates after Chi-squared test
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
