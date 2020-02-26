# PoPETs2020
Source code of the evaluations performed in Patrick Ah-Fat and Michael Huth's PoPETs'20 paper:
"Protecting Private Inputs: Bounded Distortion Guarantees With Randomised Approximations"

This repository contains the code that we used to perform our experiments on randomised approximations with bounded distortion guarantees. 

- entropy.py contains the implementation of the min-entropy H(Y|f(Y,Z)) given function f and prior distributions for Y and Z
- entropy_gen.py contains the implementation of the g-entropy H_g(Y|f(Y,Z))
- entropy_priorO.py computes the prior distribution of the output of a function f(Y,Z)
- merge_MPC.py contains the implementation of our algorithms 1 and 2. We implement the output randomisation functions h. Such function is implemented as function dmerge for algorithm 1 and as function dyn_merge for algorithm 2. 

- exp1.py to exp4.py contain our experiments 1 to 4
- exp1utility.py to exp4utility.py evaluate the utility of our mechanisms in terms of expected loss
- exp5.py and exp6.py contain additional experiments 5 and 6 on interactive functions

- experiments and experimentsUtility are directories for output files
