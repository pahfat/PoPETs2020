#!/usr/bin/python3

"""
Empirical privacy evaluation for experiment E1
"""

import sys
import entropy
from entropy_gen import wme_gen
from scipy.optimize import brute
import time
import merge_MPC
from utilities import *


############### set parameters ###############

# parse potential command line arguments
(maxi, nb_iter, save) = parseArgs(maxi_default=20, nb_iter_default=1000, save=True)

# maximal distortion
d = 1

# input domains
mini = 0
domain = range(mini, maxi + 1)

# name
nameExp = "EXP_FOUR"

# function generator, draws function fun from a family as described in the paper
def gen_f_priors():
  # function definition
  mx = 20
  mn = 1
  a,b,c,d,e,f,g,h,i = (random.randint(mn, mx) for ii in range(9))
  fun = lambda x, y, w, z: a*y*y + b*w*w + c*z*z + d*y*w + e*y*z + f*w*z + g*y + h*w + i*z
  
  # priors definition
  pYs = [r_dist(domain)]
  pZs = [r_dist(domain)]*2
  return (fun, pYs, pZs)

# domain for additive randomisations
rdomain = range(-d, d + 1)

# uniform noise
pR = uni(rdomain)

# truncated discrete laplace noise
pp = 0.3
pL = trunc_sym_geom(pp, d)


############### compute H(Y|O) ###############

# and compute H(Y|O') for randomised and truncated output

print("Computing H(Y|O)")
xfs = []
yfs = [] # original
ygs = [] # add uniform random
yhs = [] # truncation
yis = [] # for optimal # optional ...
yjs = [] # det merge
yks = [] # dyn merge
yls = [] # add truncated Lagrange
distributions = []
t = time.time()
for iii in range(nb_iter):
  pprint("[{}%]".format(round(100*iii/nb_iter, 1)))
  
  ## generate function f and its approximations
  
  # sample one function/distributions as defined above
  (f, pYs, pZs) = gen_f_priors()
  
  # gain function
  gf = {w:{y:(y[0]%2 == w) for y in iproduct(*pYs)} for w in {0,1}} # parity
  
  # pZs2 for uniform additive randomisation
  pZs2 = pZs + [pR]
  # pZs3 for truncated Lagrange random
  pZs3 = pZs + [pL]
  
  g = gen_g(f)
  h = gen_h(f, d)
  X = (999,) # attackers' input is a placeholder here: attacker is deemed to be external
  support = [f(*(X+Y+Z)) for Y in iproduct(*pYs) for Z in iproduct(*pZs)]
  j1 = merge_MPC.dmerge(support, d)
  def j(*x):
    return j1(f(*x))
  
  do = merge_MPC.do(f, pYs, pZs, X)
  k1 = merge_MPC.dyn_merge(do, d)
  def k(*x):
    return k1(f(*x))
  
  
  ## compute H(Y|O) and corresponding H(Y|O')'s
  
  xfs.append(iii)
  e1 = wme_gen(f, pYs, pZs, X, -1, gf)
  yfs.append(e1)
  e2 = wme_gen(g, pYs, pZs2, X, -1, gf)
  ygs.append(e2)
  e3 = wme_gen(h, pYs, pZs, X, -1, gf)
  yhs.append(e3)
  
  e5 = wme_gen(j, pYs, pZs, X, -1, gf)
  yjs.append(e5)
  e6 = wme_gen(k, pYs, pZs, X, -1, gf)
  yks.append(e6)
  e7 = wme_gen(g, pYs, pZs3, X, -1, gf)
  yls.append(e7)
  
  # compute optimal randomisation with independent noise
  optimal = True
  if optimal:
    # compute optimal phi, ranged in [[-d, d]] and corresponding entropy
    def objective(phi00):
      phis = [phi00.item(k) for k in range(2*d)]
      if any([x < 0 or x > 1 for x in phis + [sum(phis)]]): 
        return 0 # used if finish != None in brute
      phim1 = 1 - sum(phis)
      pPhi = {k - d: phis[k] for k in range(2*d)}
      pPhi[d] = phim1
      pPhis = [pPhi]
      pZsp = pZs + pPhis
      e = wme_gen(g, pYs, pZsp, X, -1, gf)
      return -e
      
    precisi = 6
    res = brute(objective, ((0,1),)*(2*d), full_output=True, Ns=precisi)
    e4 = -res[1]
    yis.append(e4)
    # reconstruct optimal distribution
    dist = {i - d: res[0][i] for i in range(2*d)}
    dist[d] = 1 - sum(res[0])
    distributions.append(dist)
    # }

print("\nElapsed time : {} seconds. ".format(round(time.time()-t)))

############### analysis ###############

do_sort = True
if do_sort:
  # sort by original entropy
  if optimal:
    l = zip(yfs, ygs, yhs, yis, yjs, yks, yls)
  else:
    l = zip(yfs, ygs, yhs, yjs, yks, yls)
  l = sorted(l)#, key=lambda x: x[0])
  if optimal:
    [yfs, ygs, yhs, yis, yjs, yks, yls] = zip(*l)
  else:
    [yfs, ygs, yhs, yjs, yks, yls] = zip(*l)


print("Analysing results ...")
analyse(xfs, yfs, ygs, yhs, yis, yjs, yks, yls, optimal, nameExp, save=save)

print("Plotting first values ...")
plot_values(xfs, yfs, ygs, yhs, yis, yjs, yks, yls, optimal, nameExp, "entropy")


print("Program terminated successfully.")
