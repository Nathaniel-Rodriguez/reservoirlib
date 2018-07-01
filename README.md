# reservoirlib
A library that has various tools for making, testing, and training reservoirs 
computers and echo state networks. It was developed for Python 3 and tested on
Python 3.5+.

### Library contents
#### distribution
Contains a class that wraps around Numpy's RandomState. You can use it with tasks
and reservoirs that need random numbers drawn for them.
#### esn
Contains the echo state network (ESN) classes and activation functions.
#### experiment
Contains an experiment class that uses ESNs, trainers, and tasks, to run an
experiment. An experiment involves the generation of data from
the tasks and the generation of output weights for the ESN given a cost
function defined by the trainer. It also offers validation methods defined
by the task to evaluate the trained model. Another experiment class is available
for evaluating the dynamical behavior of a reservoir given a task without 
training it.
#### generator
Contains functions for generating reservoirs and input weights for reservoirs.
It does not generate reservoir graphs from scratch. The connectivity of the graph
can be given as an edge list or adjacency matrix.
#### metric
Contains classes that can be used to evaluate the dynamical behavior of an ESN.
#### task
A series of tasks that an ESN might perform. Currently there is the Nbit memory
task and Memory Capacity task. These classes generate input and target signals
as well as define methods for validating model target signals.
#### trainer
Contains classes that define the cost function that will be optimized along with
the optimizer itself.
#### utility
Contains some commonly used functions and default parameters. It also has a 
function for running experiments using MPI. This requires mpi4py to be installed.
It is an optional dependency.
### Making New Additions
Each class has an abstract base class which defines an interface for the class.
New trainers, generators, ESNs, or tasks can be created and added
at your leisure by inheriting from these base classes.

### Using reservoirlib
An example script is provided under `examples` which you can inspect to see how
the different library elements come together to make an experiment. The examples
provided require the [graphgen library](https://github.com/Nathaniel-Rodriguez/graphgen.git)
 for generating graphs for the reservoirs. You can swap it out for your own.