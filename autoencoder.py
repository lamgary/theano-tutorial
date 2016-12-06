# http://deeplearning.net/tutorial/dA.html#daa

from __future__ import print_function

import numpy
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

try:
    import PIL.Image as Image
except ImportError:
    import Image


class DenoisingAutoEncoder:
    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            input=None,
            n_visible=784,
            n_hidden=500,
            W=None,
            bhid=None,
            bvis=None
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        #Theano random generator
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.theano_rng = theano_rng

        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low = -4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high = 4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size = (n_visible, n_hidden)
                ),
                dtype = theano.config.floatX
            )

            W = theano.shared(value=initial_W, name='W', borrow=True)

        # ?? why shared variables between layers?
        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W

        self.b = bhid

        self.b_prime = bvis
        self.W_prime = self.W.T

        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
               same and zero-out randomly selected subset of size ``coruption_level``
               Note : first argument of theano.rng.binomial is the shape(size) of
                      random numbers that it should produce
                      second argument is the number of trials
                      third argument is the probability of success of any trial

                       this will produce an array of 0s and 1s where 1 has a
                       probability of 1 - ``corruption_level`` and 0 with
                       ``corruption_level``

                       The binomial function return int64 data type by
                       default.  int64 multiplicated by the input
                       type(floatX) always return float64.  To keep all data
                       in floatX when floatX is float32, we set the dtype of
                       the binomial to floatX. As in our case the value of
                       the binomial is always 0 or 1, this don't change the
                       result. This is needed to allow the gpu to work
                       correctly as it only support float32 for now.

        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p = 1 - corruption_level,
                                        dtype=theano.config.floatX) * input


    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer"""
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Reconstruct input given hidden layer"""
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """Computes cost and updates for 1 training step of dA"""

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        #sum over size of a datapoint, one entry per example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log( 1- z), axis = 1)

        cost = T.mean(L)

        # compute gradients of cost of 'dA' w.r.t. parameters
        gparams = T.grad(cost, self.params)

        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return(cost, updates)


