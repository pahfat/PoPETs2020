#!/usr/bin/python3

"""
Function dmerge() implements Algorithm 1
Function dyn_merge() implements Algorithm 2
"""

import itertools
import operator
from functools import reduce

def prod(X):
  return reduce(operator.mul, X, 1)

# takes an output support O
# and returns a function 
# that maps O to a new domain O'
# dmerge (deterministic merge) merges as many outputs as
# maximal distortion d allows, from lowest output
def dmerge(support, max_d): 
  y = min(support) - max_d - 1
  d = dict()
  for x in sorted(support):
    if x - y > max_d:
      y = x + max_d
    d[x] = y
  def f(x):
    return d[x]
  return f


# do[o] : max_y p(y).p(o|y)
def dyn_merge(do, delta):
  no = len(do)
  S = [0 for i in range(no + 1)]
  N = [0 for i in range(no + 1)]
  A = [0] + sorted(do.keys())
  jmin = 1
  for j in range(1, no + 1):
    while A[j] - A[jmin] > 2*delta: 
      jmin += 1
    m = do[A[j]]
    smin = m + S[j - 1]
    nmin = j
    for i in range(j - 1, jmin - 1, -1):
      m = max(m, do[A[i]])
      s = m + S[i - 1]
      if s < smin:
        smin = s
        nmin = i
    S[j] = smin
    N[j] = nmin
  # reconstruct output
  j = no
  h = dict()
  while j > 0:
    n = N[j]
    op = (A[n] + A[j])//2
    for i in range(n, j + 1):
      h[A[i]] = op
    j = n - 1
  # construct function on same domain as do
  return lambda o: h[o]


# produce do s.t.: do[o] = max_y p(y) p(o | y)
# very similar to wme algorithm
def do(f, pYs, pZs, X):
  d = dict()
  for Y in itertools.product(*pYs):
    for Z in itertools.product(*pZs):
      o = f(*(X+Y+Z))
      po = prod( (pYZ[yz] for (yz, pYZ) in zip(Y+Z, pYs+pZs)) )
      # critical for distributions with a zero probability!
      # next line ensures that only possible events appear in pO (the if)
      if po > 0:
        if o in d:
          d[o] = max(d[o], po)
        else:
          d[o] = po
  return d
