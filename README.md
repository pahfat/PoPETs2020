# Protecting Private Inputs: Bounded Distortion Guarantees With Randomised Approximations

This repository contains the code that we used to perform our experiments on randomised approximations with bounded distortion guarantees. 
Its objective is to enable a reader to reproduce the results of the experiments presented in our PoPETs'2020 paper. 


## Summary of paper and experiments

Secure Multi-Party Computation (SMC) is a domain of cryptography that enables several parties to compute a public function of their private inputs, while keeping their inputs secret and without having to rely on any other trusted third party. Advanced protocols have been designed in order to achieve this functionality while ensuring that no information leaks about the private inputs apart from the intended public output. 
Paradoxically, as a function of the inputs, the output of such computation inevitably reveals *some* information about private inputs, regardless of how secure the implementation is. This is commonly referred to the *acceptable leakage* and is therefore largely ignored in the literature on SMC. 

We believe that this information leakage may not always be acceptable and we develop randomising mechanisms that aim at enhancing the privacy of inputs in SMC. 
Our work relies on two fundamental assumptions that also distinguishes itself from Differential Privacy:
- privacy is measured by means of entropy-based measures, stemming from Quantitative Information Flow. 
- our randomisations offer hard guarantees that the distorted outputs will be contained within a certain distortion threshold. 

We theoretically investigate the privacy gains that randomisations can provide based on those two assumptions and we further explore the case of so called *sparse* functions. 
We also design two randomising mechanisms `Gdet` and `Gdyn` and we prove that they maximise the inputs' privacy in the secure computation of sparse functions under uniform and non-uniform prior beliefs on the inputs respectively. 

In our experiments, we compare our algorithms to other existing randomising mechanisms that guarantee hard distortion bounds. 
We summarise the aims of the present experiments as follows:
- We evaluate the privacy gains of our algorithms on sparse functions, where we expect our algorithm `Gdyn` to be optimal (with Experiments 1 and 2). 
- We demonstrate that our algorithms also provide high privacy gains on non-sparse functions compared to existing mechanisms (with Experiments 3 and 4). 
- We demonstrate that high privacy gains are also achieved by our algorithms on non-polynomial functions (with Experiments 5 and 6). 
- Finally, we evaluate the utility of our algorithms in terms of *expected gain*, where we expect our algorithms to perform worse than the truncated Laplace mechanism `Glap` whose bespoke construction ensures high utility guarantees (with utility experiments). Surprisingly, we notice that `Gdyn` still performs better than `Glap` for Experiment 1. 


## Files description

- Core helper functions 
  - *entropy.py* contains the implementation of the min-entropy `H(Y|f(Y,Z))` given function f and prior distributions for Y and Z. We first browse all input combinations to compute the prior probability distribution `p(O)` and the joint distribution `p(O,Y)` where `O = f(Y,Z)` denotes the output. We then compute the posterior distribution `p(Y|O)` as `p(O,Y)/p(O)` and output its entropy. 
  - *entropy_gen.py* contains the implementation of the g-entropy `Hg(Y|f(Y,Z))`. Implementation details are similar to that of min-entropy. 
  - *entropy_priorO.py* computes the prior distribution of the output of a function `f(Y,Z)`. Implemenation details are similar to that of min-entropy. 
  - *merge_MPC.py* contains the implementation of our algorithms 1 and 2. We implement the output randomisation functions h. Such function is implemented as function **dmerge** for algorithm 1 and as function **dyn_merge** for algorithm 2. Implementation details can be found in the paper. 

- Experiments
  - *exp1.py* to *exp4.py* contain the code that implements our experiments 1 to 4. For each of the **nb_iter** iterations, we generate a function f from a family of functions that is explicitly defined in the paper. We define its different randomisations and we compare the privacy that they provide. 
  - *exp1utility.py* to *exp4utility.py* evaluate the utility of our mechanisms in terms of expected loss. 
  - *exp5.py* and *exp6.py* contain additional experiments 5 and 6 on interactive functions. 

- Others
  - *utilities.py* contains functions that help running the experiments such as parsing, setting the parameters, plotting and saving results. 
  - *experiments* and *experimentsUtility* are directories for output files. 
  - *requirements.txt* contains the Python modules required to run the code and the versions that were used to perform the tests


## How to run the experiments

### Dependencies

This project requires Python3 installed with the following modules: mpmath, numpy, scipy and matplotlib. The experiments have been run with Python 3.6.9 and the versions of the required modules are included in *requirements.txt*, which is the output produced by pipreqs for this project (dependencies can be installed by running `pip3 install -r requirements.txt`). 

### Usage

Experiments (Python files starting with *exp*) can be run by executing the corresponding script in the following ways:
- by specifying the maximal input size **maxi** and the number of iterations **nb_iter** respectively as two positive integers (optionally followed by the 'nosave' flag if data need not be saved as a pickle object)
- without any argument, in which case the experiment will be run with defaut values corresponding to the experiments run in our paper. The average values of entropy and utility output by this configuration are the ones that are reported in the paper. As mentioned below, this may take a few hours, so we suggest to try the previous approach first with smaller parameters. We also suggest to set **nb_iter** to 20 and **maxi** to the default value to produce graphical representations that are similar to those presented in the paper. 

For example, you can set **maxi** to 4 and **nb_iter** to 6 and run Experiment 1 by running:

> ./exp1.py 4 6

Similarly if you do not with to save the results into a pickle object, run:

> ./exp1.py 4 6 nosave

If you wish to perform the utility evaluation of Experiment 3 with the default parameters that were used in our paper, run:

> ./exp3utility.py

Similarly, for testing purposes, you can set **maxi** to 5 and **nb_iter** to 10 for this experiment with:

> ./exp3utility.py 5 10


The experiments described in our paper were run on a machine with an Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz processor. The approximate time that each experiment took to complete is given below for information purposes:
- exp1.py       : 12h
- exp2.py       : 12h
- exp3.py       : 5h
- exp4.py       : 10h
- exp5.py       : 14h
- exp6.py       : 20h
- exp1utility.py: 12h
- exp2utility.py: 12h
- exp3utility.py: 5h
- exp4utility.py: 2h


## Acknowledgments

We would like to thank anonymous reviewers who helped us to improve the readability and usability of our code. 


## License

This project is licensed under the MIT license. 
