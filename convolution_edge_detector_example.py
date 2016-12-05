import numpy
import pylab
from PIL import Image

import convolution

img = Image.open(open('3wolfmoon.jpg'))

#Dimensions are (height, width, channel)
img = numpy.asarray(img, dtype='float64') / 256

#Put image into 4D tensor of shape (1,3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1,3,639,516)

filtered_img = convolution.f(img_)

#plot each axis

pylab.subplot(1,3,1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
pylab.subplot(1,3,2);pylab.axis('off'); pylab.imshow(filtered_img[0,0,:,:])
pylab.subplot(1,3,3);pylab.axis('off'); pylab.imshow(filtered_img[0,1,:,:])
pylab.show()