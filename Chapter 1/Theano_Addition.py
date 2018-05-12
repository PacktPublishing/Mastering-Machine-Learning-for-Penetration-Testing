from theano import *
import theano.tensor as T
from theano import function
a = T.dscalar('a')
b = T.dscalar('b')
c = a + b
f = function([a, b], c)
print (f)
