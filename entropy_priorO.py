#!/usr/bin/python3

"""
Function priorO() computes the prior probability of output
"""

# notations

# X : attackers (vector)
# Y : victims (vector)
# Z : other parties (vector)
# CAPITAL letters denote vectors

import itertools
from functools import reduce
import operator
import inspect
import sys
import math

iproduct = itertools.product

def prod(X):
  return reduce(operator.mul, X, 1)

def min_entropy(S):
  return - math.log2(max(p for p in S))

# computes prior output distribution
# f : function
# pYs : prior distributions for the Ys (list of dictionaries)
# pZs : prior distributions for the Zs (list of dictionaries)
# X : a given vector X (as a tuple)
def priorO(f, pYs, pZs, X):
  # precondition
  # bypass precondition if number of arguments == 0, i.e. if f is defined as a function of another function
  if len(X) + len(pYs) + len(pZs) != len(inspect.getargspec(f)[0]) and len(inspect.getargspec(f)[0]) != 0:
    sys.exit("Aborted: Number of function arguments and inputs didn't match")
  
  if len(inspect.getargspec(f)[0]) == 0:
    try:
      a = f(*((0,)*(len(X) + len(pYs) + len(pZs))))
    except:
      sys.exit("Aborted: Number of function arguments and inputs didn't match")
  
  # prior distribution for Output
  pO = dict()
  # joint distribution for Output and Y
  pOaY = dict()
  for Y in itertools.product(*pYs):
    for Z in itertools.product(*pZs):
      o = f(*(X+Y+Z))
      p_o = prod( (pYZ[yz] for (yz, pYZ) in zip(Y+Z, pYs+pZs)) )
      # critical for distributions with a zero probability!
      # next line ensures that only possible events appear in pO (the if)
      if p_o > 0:
        if o in pO:
          pO[o] += p_o
          if Y in pOaY[o]:
            pOaY[o][Y] += p_o
          else:
            pOaY[o][Y] = p_o
        else:
          pO[o] = p_o
          pOaY[o] = {Y: p_o}
        
  return pO

