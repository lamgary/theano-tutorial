import cPickle, gzip, numpy

# Load the dataset
def load(path):
  f = gzip.open('mnist.pkl.gz', 'rb')
  train_set, valid_set, test_set = cPickle.load(f)
  f.close()
  return(train_set, valid_set, test_set)
