{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled9.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "ePTOM_MBbfPb",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "\n",
        "import pyopencl as cl\n",
        "import numpy as np\n",
        "ASTRO_OBJ_SPEC_SIZE = 4096\n",
        "\n",
        "\n",
        "#To calculate simple moving average\n",
        "def simple_mavg(QsoCL,inpMat, window):\n",
        "    queue = cl.CommandQueue(QsoCL.ctx)\n",
        "    \n",
        "    h = inpMat.shape[0]  # spectrumsSize\n",
        "    w = inpMat.shape[1]  # spectrumsNumber\n",
        "    globalSize = QsoCL.calcGlobalSize(w)\n",
        "    res_np = np.zeros((h, w), dtype=np.float64)\n",
        "    a_g =  QsoCL.readBuffer(inpMat)\n",
        "    res_g = QsoCL.writeBuffer(res_np)\n",
        "\n",
        "    _knl = QsoCL.buildKernel('mavg_kernels.cl').simple_mavg\n",
        "\n",
        "    _knl.set_scalar_arg_dtypes([None, np.uint32, np.uint32, None, np.uint32])\n",
        "    _knl(queue, (globalSize,), (QsoCL.maxWorkGroupSize,),\n",
        "         a_g, w, h, res_g, window)\n",
        "    cl.enqueue_copy(queue, res_np, res_g)\n",
        "    #res_np = np.transpose(res_np)\n",
        "    return res_np\n",
        "\n",
        "#To calculate centered moving average\n",
        "def centered_mavg(QsoCL,inpMat, sizes, window):\n",
        "    queue = cl.CommandQueue(QsoCL.ctx)\n",
        "    \n",
        "    h = inpMat.shape[0]  # spectrumsSize\n",
        "    w = inpMat.shape[1]  # spectrumsNumber\n",
        "    globalSize = QsoCL.calcGlobalSize(w)\n",
        "    res_np = np.zeros((h, w), dtype=np.float64)\n",
        "\n",
        "    a_g = QsoCL.readBuffer(inpMat)\n",
        "    bufferSizes = QsoCL.readBuffer(sizes)\n",
        "    res_g = QsoCL.writeBuffer(res_np)\n",
        "\n",
        "    _knl = QsoCL.buildKernel('mavg_kernels.cl').centered_mavg\n",
        "    _knl.set_scalar_arg_dtypes(\n",
        "        [None, np.uint32, np.uint32, None, None, np.uint32])\n",
        "    _knl(queue, (globalSize,), (QsoCL.maxWorkGroupSize,),\n",
        "         a_g, w, h, bufferSizes, res_g, window)\n",
        "    cl.enqueue_copy(queue, res_np, res_g)\n",
        "    return res_np\n"
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}
