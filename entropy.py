#!/usr/bin/python3

"""
function wme() implements the min-entropy H(Y|f(Y,Z))
"""

# notations

# X : attackers (vector)
# Y : targets (vector)
# Z : spectators (vector)

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


# f : function
# pYs : prior distributions for the Ys (list of dictionaries)
# pZs : prior distributions for the Zs (list of dictionaries)
# X : a given vector X (as a tuple)
def wme(f, pYs, pZs, X):
  # precondition
  # bypass precondition if number of arguments == 0, i.e. if f is defined as a function of another function
  if len(X) + len(pYs) + len(pZs) != len(inspect.getargspec(f)[0]) and len(inspect.getargspec(f)[0]) != 0:
    sys.exit("Aborted: Number of function arguments and inputs didn't match")
  
  if len(inspect.getargspec(f)[0]) == 0:
    try:
      a = f( *( X + tuple(list(pp.keys())[0] for pp in pYs) + tuple(list(pp.keys())[0] for pp in pZs)) )
    except KeyError:
      # here we tolerate KeyError because merged functions might not recognise input (0, 0, ...)
      pass
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
        
  # then convert pOaY into pYgO (kinda Bayes)
  # maxsY stores the maximal probability of each distribution of pYgO for each o
  maxsY = dict()
  for output in pO: # or in pOgY, equivalently
    for Y in pOaY[output]:
      pOaY[output][Y] /= pO[output]
      
    maxsY[output] = max(pOaY[output].values())
    
  # weighted average entropy
  wme = - math.log2( sum(pO[output]*maxsY[output] for output in pO) )
  
  return wme

