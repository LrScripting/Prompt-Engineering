# requires filename and flat list of convolution kernal to be input in as a Cli arg.
import argparse
import numpy as np
import scipy.signal
from PIL import Image

class Convolution:
    def __init__(self, image, stride=1, padding=0):
        self.image = image
        self.stride = stride
        self.padding = padding

    def convolve(self, kernel):
        ksize = kernel.shape[0]
        return self._convolve_operation(self.image, kernel, ksize)

    def _convolve_operation(self, image, kernel, ksize):
        if len(image.shape) > 2:
            output_channels = []
            for channel in range(image.shape[2]):
                output_channels.append(scipy.signal.convolve2d(image[:, :, channel], kernel, mode='valid'))
            return np.stack(output_channels, axis=-1)
        else:
            return scipy.signal.convolve2d(image, kernel, mode='valid')

def main(image_path, kernel, stride=1, padding=0):
    image = Image.open(image_path)
    image_data = np.asarray(image)

    if len(image_data.shape) > 2:
        image_data = np.dot(image_data[...,:3], [0.2989, 0.5870, 0.1140])

    conv = Convolution(image_data, stride, padding)
    result = conv.convolve(np.array(kernel))
    
    # Print the resulting image data and shape
    print(result)
    print(result.shape)
    img = Image.fromarray(result)
    img.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform a convolution operation on an image.")
    parser.add_argument("image_path", help="The path to the image.")
    parser.add_argument("kernel", nargs='+', type=float, help="The kernel to use for the convolution.")
    parser.add_argument("--stride", default=1, type=int, help="The stride to use for the convolution.")
    parser.add_argument("--padding", default=0, type=int, help="The padding to use for the convolution.")

    args = parser.parse_args()
    
    # Converting the kernel input into 2D list.
    kernel_size = int(np.sqrt(len(args.kernel)))
    kernel = np.array(args.kernel).reshape(kernel_size, kernel_size)

    main(args.image_path, kernel, args.stride, args.padding)
