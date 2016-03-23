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
    * memory: symbolic program is compiled into memory graph, by knowing when a varible is finished using, can release resource immediately. Comparing to imperative releases after out-of-scope. _Will it be a big deal? If it's a big deal I think we can invent something doing more strict out-of-scope anlysis. (e.g., insert more fine grained scope into code)._
    * memory: symbolic program is good at defining/inferring computation boundaries (e.g., when only need forward pass, don't need backward pass, do not need to store gradient used for backward propagation). _In the example case, imperative program can do it as well. I can feel that by using symbolic program and building graphs can optimize memory storage by ignore storing the varibles not needed for the computation in latter stage. But I have not yet see the case that imperative program can not do and have a significant memory improvement that will justify the claim._
    * runtime: symobic program can do operation folding: do `a * b + 1` in one single GPU kernel instead two separate ones. _I wonder if GPU compilers like nvcc does it automatically?_
