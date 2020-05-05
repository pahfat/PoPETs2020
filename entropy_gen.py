#!/usr/bin/python3

"""
function wme_gen() implements the g-entropy H_g(Y|f(Y,Z))
"""

# notations

# X : attackers (vector)
# Y : victims (vector)
# Z : other parties (vector)

import itertools
from functools import reduce
import operator
import inspect
import sys
import mpmath

iproduct = itertools.product

def prod(X):
  return reduce(operator.mul, X, 1)

def p_norm(xs, p):
  return sum(x**p for x in xs)**(1/p)

# we don't use renyi entropy here, but will use it outside (for Theorem 2)
def renyi_entropy(l, alpha):
  ll = [mpmath.mpmathify(x) for x in l]
  if alpha == -1:
    return - mpmath.log(max(l), b=2)
  else:
    return mpmath.log(sum(x**alpha for x in ll), b=2) / (1 - alpha)

# alpha : real in ]0, 1[ union ]1, infinity[ (alpha=1 is special)  ### -1 for infinity
# pYs, pZs : list of dictionaries
# g: gain function, as dictionary of dictionary: g[w][y] = gain
# optional delta, component of offset vector to add to W before norm
def wme_gen(f, pYs_p, pZs_p, X, alpha, g_p, delta=0):
  # convert probabilities to mpmath.mpf
  pYs = [{k: mpmath.mpmathify(pY[k]) for k in pY} for pY in pYs_p]
  pZs = [{k: mpmath.mpmathify(pZ[k]) for k in pZ} for pZ in pZs_p]
  g = {w:{y:mpmath.mpmathify(g_p[w][y]) for y in g_p[w]} for w in g_p}
  delta = mpmath.mpmathify(delta)
  
  # precondition
  # bypass precondition if number of arguments == 0, i.e. if f is defined as a function of another function
  if len(X) + len(pYs) + len(pZs) != len(inspect.getargspec(f)[0]) and len(inspect.getargspec(f)[0]) != 0:
    sys.exit("Aborted: Number of function arguments and inputs didn't match")
  
  if len(inspect.getargspec(f)[0]) == 0:
    try:
      a = f(*((0,)*(len(X) + len(pYs) + len(pZs))))
    except KeyError:
      # here we tolerate KeyError because merged functions might not recognise input (0, 0, ...)
      pass
    except:
      sys.exit("Aborted: Number of function arguments and inputs didn't match")
  
  # joint distribution for Output and Y
  pOaY = dict()
  for Y in itertools.product(*pYs):
    for Z in itertools.product(*pZs):
      o = f(*(X+Y+Z))
      p_o = prod( (pYZ[yz] for (yz, pYZ) in zip(Y+Z, pYs+pZs)) )
      # critical for distributions with a zero probability!
      # next line ensures that only possible events appear in pO (the if)
      if p_o > 0:
        if o in pOaY:
          if Y in pOaY[o]:
            pOaY[o][Y] += p_o
          else:
            pOaY[o][Y] = p_o
        else:
          pOaY[o] = {Y: p_o}
        
  # W[o] is the vector inside the norm (over guesses w)
  W = dict()
  for o in pOaY: 
    W[o] = [sum(pOaY[o][Y]*g[w][Y] for Y in g[w] if Y in pOaY[o]) + delta for w in g]
  
  if alpha == -1:
    sum_o = sum(max(W[o]) for o in pOaY)
    result = - mpmath.log(sum_o, b=2)
  else:
    sum_o = sum(p_norm(W[o], alpha) for o in pOaY)
    result = (alpha/(1-alpha))*mpmath.log(sum_o, b=2)
  
  return result

