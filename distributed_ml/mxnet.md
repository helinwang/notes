Notes about MXNet while reading:

[MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems](http://www.cs.cmu.edu/~muli/file/mxnet-learning-sys.pdf)

[MXNet: Programming Models for Deep Learning](http://mxnet.readthedocs.org/en/latest/program_model.html)

* Programming model of different NN libraries: `symbolic` vs `imperative`
  * symbolic:
  ```
A = Variable('A')
B = Variable('B')
C = B * A
D = C + Constant(1)
# compiles the function
f = compile(D)
d = f(A=np.ones(10), B=np.ones(10)*2)
  ```
  * imperative:
  ```
import numpy as np
a = np.ones(10)
b = np.ones(10) * 2
c = b * a
d = c + 1
  ```
  * Imperative Programs are More Flexible
    * because programming languages itself tends to support more syntax than symbolic language
  * Symbolic Programs are More Efficient
    * memory: symbolic program is compiled into memory graph, by knowing when a varible is finished using, can release resource immediately. Comparing to imperative releases after out-of-scope. Will it be a big deal?
    
