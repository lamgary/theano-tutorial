import load_mnist
import logistic_regression as lr 
from theano import tensor as T

mnist_file = 'mnist.pkl.gz'

y = T.ivector('y')
x = T.matrix('x')
classifier = lr.LogisticRegression(input=x, n_in=28 * 28, n_out = 10)

cost = classifier.negative_log_likelihood(y)

g_W = T.grad(cost=cost, wrt=classifier.W)
g_b = T.grad(cost=cost, wrt=classifier.b)

updates = [(classifier.W, classifier.W - learning_rate * g_W),
           (classifier.b, classifier.b - learning_rate * g_b)]

train_model = theano.function(
  inputs=[index],
  outputs=cost,
  updates=updates,
  givens={
      x: train_set_x[index * batch_size: (index + 1) * batch_size],
      y: train_set_y[index * batch_size: (index + 1) * batch_size]
  }
)

test_model = theano.function(
  inputs=[index],
  outputs=classifier.errors(y),
  givens={
      x: test_set_x[index * batch_size: (index + 1) * batch_size],
      y: test_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

validate_model = theano.function(
    inputs=[index],
    outputs=classifier.errors(y),
    givens={
        x: valid_set_x[index * batch_size: (index + 1) * batch_size],
        y: valid_set_y[index * batch_size: (index + 1) * batch_size]
    }
)

(train_set, valid_set, test_set) = load_mnist.load(mnist_file)
