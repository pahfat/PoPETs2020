#!/usr/bin/python3

"""
Empirical privacy evaluation for experiment E1
"""

import sys
import entropy
from scipy.optimize import brute
import time
import merge_MPC
import entropy_priorO
from utilities import *


############### set parameters ###############

# parse potential command line arguments
(maxi, nb_iter, save) = parseArgs(maxi_default=30, nb_iter_default=1000, save=True)

# maximal distortion
d = 1

# input domains
mini = 0
domain = range(mini, maxi + 1)

# name
nameExp = "EXP_THREE"

# function generator, draws function fun from a family as described in the paper
def gen_f_priors():
  mx = 20
  mn = 1
  a,b,c,d,e,f = (random.randint(mn, mx) for ii in range(6))
  fun = lambda x, y, z, t: a*y + b*z + c*t + d*a*z + e*a*t +  f*z*t
  
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

print("Computing utilities")
xfs = []

# utilities
ug = [] # add uniform random
uh = [] # truncation
ui = [] # for optimal
uj = [] # det merge
uk = [] # dyn merge
ul = [] # add truncated Lagrange

u_lag = pL[0]
u_uni = pR[0]

distributions = []
t = time.time()
for iii in range(nb_iter):
  pprint("[{}%]".format(round(100*iii/nb_iter, 1)))
  
  ## generate function f and its approximations
  
  # sample one function/distributions as defined above
  (f, pYs, pZs) = gen_f_priors()
  
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

  
  xfs.append(iii)
  
  # utility computation!
  pO = entropy_priorO.priorO(f, pYs, pZs, X)
  
  # constants
  ug.append(u_uni)
  ul.append(u_lag)
  
  # for deterministic merge
  u_det = sum(pO[o]*(j1(o) == o) for o in pO)
  uj.append(u_det)
  #print("u_det = ", u_det)
  
  # for dynamic merge  
  u_dyn = sum(pO[o]*(k1(o) == o) for o in pO)
  uk.append(u_dyn)
  #print("u_dyn = ", u_dyn)
  
  # for truncation  
  u_trunc = sum(pO[o]*((o - (o % (2*d + 1)) + d) == o) for o in pO)
  uh.append(u_trunc)
  
  
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
      e = entropy.wme(g, pYs, pZsp, X)
      return -e
      
    precisi = 6
    res = brute(objective, ((0,1),)*(2*d), full_output=True, Ns=precisi)

    # reconstruct optimal distribution
    dist = {i - d: res[0][i] for i in range(2*d)}
    dist[d] = 1 - sum(res[0])
    distributions.append(dist)
    # }
    
    u_opt = dist[0]
    ui.append(u_opt)

print("\nElapsed time : {} seconds. ".format(round(time.time()-t)))

############### analysis ###############


print("Analysing results ...")
analyse(xfs, None, ug, uh, ui, uj, uk, ul, optimal, nameExp, save=save, utility=True)

print("Plotting first values ...")
plot_values(xfs, None, ug, uh, ui, uj, uk, ul, optimal, nameExp, "utility", utility=True)


print("Program terminated successfully.")
