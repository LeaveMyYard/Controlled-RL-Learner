import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import numpy.linalg as la
from pycuda.compiler import SourceModule

import time

mod = SourceModule(open("cuda/grayscale.cu").read())

dot = mod.get_function("dot")


def grayscale(image):
    a = image.astype(np.int32)
    image_shape = image.shape

    result = np.ones((image_shape[0], image_shape[1]), dtype=np.int32)
    dot(
        drv.Out(result),
        drv.In(a),
        block=(1, 1, 1),
        grid=(image_shape[0], image_shape[1]),
    )

    return result


if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt

    im = np.array(Image.open("file.jpg"))

    b = grayscale(im)

    plt.imshow(im)
    plt.show()

    plt.imshow(b)
    plt.show()
