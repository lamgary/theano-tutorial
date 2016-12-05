import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d

import numpy

rng = numpy.random.RandomState(23455)


#4D tensor input: [ mini-batch-size * 3 [RGB] * image height * image width]
input = T.tensor4(name='input')

#initialize shared variable for weights
w_shp = (2, 3, 9, 9)  #output_feature_maps, RGB, 9x9 field vision
w_bound = numpy.sqrt(3 * 9 * 9)

W = theano.shared(
        numpy.asarray(
            rng.uniform(
                low = -1.0/ w_bound,
                high = 1.0 / w_bound,
                size = w_shp
            ),
            dtype = input.dtype), name = 'W')


#initialize bias (normally set to 0, for demonstrative purposes set to non-zero)
b_shp = (2,)
b = theano.shared(
    numpy.asarray(
        rng.uniform(low=-0.5, high = .5, size=b_shp),
        dtype=input.dtype), name ='b')

#build symbolic expression that computes output
conv_out = conv2d(input, W)

output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

f = theano.function([input], output)