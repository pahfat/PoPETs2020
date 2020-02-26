#!/usr/bin/python3

"""
Empirical privacy evaluation for experiment E3
"""

# we don't use x at all and consider it as fixed
# (in the present representation)

# add parent directory to path
import sys
import entropy
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import brute
import pickle
import time
#import merge
import merge_MPC
import random
from numpy.random import dirichlet

# cartesian product
iproduct = itertools.product

def pprint(x):
  print(x, flush=True, end='')


############### maximal distortion ###############
 
d = 1

############### generate randomisations ###############

# additive approximation
def gen_g(f):
  def g(*xs):
    return f(*xs[:-1]) + xs[-1]
  return g

# truncation
def gen_h(f):
  def h(*xs):
    o = f(*xs)
    return o - (o % (2*d + 1)) + d
  return h

############### input distributions ###############

# builds a uniform distribution
def uni(domain):
  return {k:1/len(domain) for k in domain}

def tri(domain):
  n = len(domain)
  return {k:(i+1)*2/(n*(n+1)) for i,k in enumerate(domain)}

def tri_down(domain):
  n = len(domain)
  return {k:(n-i)*2/(n*(n+1)) for i,k in enumerate(domain)}

# define here and use truncated (discrete) Lagrange distribution as mentioned by reviewer
# defines random distribution on domain, drawn from Dirichlet symmetric(1)
p_dirichlet = 1
def r_dist(domain):
  d = dirichlet([p_dirichlet]*len(domain))
  return {x: d[i] for i,x in enumerate(domain)}

# truncated symmetric geometric distribution around 0, max distortion delta, parameter p in [0,1]
# cite paper
def trunc_sym_geom(p, delta):
  h = {k: p**abs(k)*(1-p)/(1+p) for k in range(-delta, delta+1)}
  s = sum(h.values())
  return {x: h[x]/s for x in h}


######################################################
############### function/distributions ###############

maxi = 30
mini = 0
domain = range(mini, maxi + 1)

# name and definition

nameExp = "EXP_THREE"

def gen_f_priors():
  # function definition
  mx = 20
  mn = 1
  a,b,c,d,e,f = (random.randint(mn, mx) for ii in range(6))
  fun = lambda x, y, z, t: a*y + b*z + c*t + d*a*z + e*a*t +  f*z*t
  
  # priors definition
  pYs = [r_dist(domain)]
  pZs = [r_dist(domain)]*2
  return (fun, pYs, pZs)




# tools for additive randomisations
rdomain = range(-d, d + 1)
# uniform noise
pR = uni(rdomain)
# truncated discrete laplace noise
pp = 0.3
pL = trunc_sym_geom(pp, d)
# use those noise distributions in loop!


# tests parameters

nb_iter = 1000



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
coef_list = []
distributions = []
t = time.time()
for iii in range(nb_iter):
  pprint("[{}%]".format(round(100*iii/nb_iter, 1)))
  
  # sample one function/distributions as defined above
  (f, pYs, pZs) = gen_f_priors()
  
  
  # pZs2 for uniform additive randomisation
  pZs2 = pZs + [pR]
  # pZs3 for truncated Lagrange random
  pZs3 = pZs + [pL]
  
  g = gen_g(f)
  h = gen_h(f)
  X = (999,) # unused
  support = [f(*(X+Y+Z)) for Y in iproduct(*pYs) for Z in iproduct(*pZs)]
  j1 = merge_MPC.dmerge(support, d)
  def j(*x):
    return j1(f(*x))
  #coef_list.append(coeffs)
  
  do = merge_MPC.do(f, pYs, pZs, X) # last argument useless
  k1 = merge_MPC.dyn_merge(do, d)
  def k(*x):
    return k1(f(*x))
  
  X = (999,) # unused
  xfs.append(iii)
  e1 = entropy.wme(f, pYs, pZs, X)
  yfs.append(e1)
  e2 = entropy.wme(g, pYs, pZs2, X)
  ygs.append(e2)
  e3 = entropy.wme(h, pYs, pZs, X)
  yhs.append(e3)
  
  e5 = entropy.wme(j, pYs, pZs, X)
  yjs.append(e5)
  e6 = entropy.wme(k, pYs, pZs, X)
  yks.append(e6)
  e7 = entropy.wme(g, pYs, pZs3, X)
  yls.append(e7)
  
  
  optimal = True
  if optimal:
    # compute optimal phi, ranged [[-1, 1]]
    # {
    # calculate max phi and corresponding entropy
    def objective(phi00):
      #phi0 = phi00.item(0)
      #phi1 = phi00.item(1)
      phis = [phi00.item(k) for k in range(2*d)]
      if any([x < 0 or x > 1 for x in phis + [sum(phis)]]): 
        return 0 # used if finish != None in brute
      phim1 = 1 - sum(phis)
      #pPhi = {-1: phim1, 0: phi0, 1: phi1}
      pPhi = {k - d: phis[k] for k in range(2*d)}
      pPhi[d] = phim1
      pPhis = [pPhi]
      pZsp = pZs + pPhis
      e = entropy.wme(g, pYs, pZsp, X)
      return -e
    #res = brute(objective, ((0,1),)*(2*d), full_output=True)#, Ns=21)
    precisi = 6
    res = brute(objective, ((0,1),)*(2*d), full_output=True, Ns=precisi)
    # add "finish=None" if no polish is needed
    #print("optimal phi0 = {}".format(res[0])) # add value for 1
    ##print("optimal entropy = {}".format(-res[1]))
    e4 = -res[1]
    yis.append(e4)
    # reconstruct optimal distribution
    dist = {i - d: res[0][i] for i in range(2*d)}
    dist[d] = 1 - sum(res[0])
    distributions.append(dist)
    # }

print("\nElapsed time : {}".format(round(time.time()-t)))

############### analysis ###############

def avg(l):
  if l==[]: return 0
  return sum(l)/len(l)

avgf = avg(yfs)
avgg = avg(ygs)
avgh = avg(yhs)
avgi = avg(yis)
avgj = avg(yjs)
avgk = avg(yks)
avgl = avg(yls)

print("avgf = {}".format(avgf))
print("avgg = {}".format(avgg))
print("avgh = {}".format(avgh))
print("avgi = {}".format(avgi))
print("avgj = {}".format(avgj))
print("avgk = {}".format(avgk))
print("avgl = {}".format(avgl))

# rest in plot section (plot first 20)


############### plot ###############

print("Plotting ...")

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

# choose how many to plot
nFirst = 20

plt.plot(xfs[:nFirst], yfs[:nFirst], 'bx-', label="f")
plt.plot(xfs[:nFirst], ygs[:nFirst], 'gx:', label="add uniform")
plt.plot(xfs[:nFirst], yhs[:nFirst], 'cx-.', label="truncation")
plt.plot(xfs[:nFirst], yjs[:nFirst], 'kx--', label="merged")
plt.plot(xfs[:nFirst], yks[:nFirst], 'rx:', label="dynamic")
plt.plot(xfs[:nFirst], yls[:nFirst], 'yx:', label="add Laplace")
if optimal:
  plt.plot(xfs[:nFirst], yis[:nFirst], 'mx-', label="optimal")

title = "awae"
plt.title(title)
plt.xlabel("iteration")
plt.ylabel("$awae$")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.savefig("results/{}.pdf".format(info), bbox_inches='tight')
plt.savefig("experiments/{}.pdf".format(nameExp), bbox_inches='tight')

############### data print ###############

text = ""

if optimal:
  text += ("x fx uni opt trunc det dyn lap\n")
else:
  text += ("x fx uni trunc det dyn lap\n")

for i in range(min(nb_iter, nFirst)):
  if optimal:
    text += ("{} {} {} {} {} {} {} {}\n".format(i, yfs[i], ygs[i], yis[i], yhs[i], yjs[i], yks[i], yls[i]))
  else:
    text += ("{} {} {} {} {} {} {}\n".format(i, yfs[i], ygs[i], yhs[i], yjs[i], yks[i], yls[i]))

print(text)

############### pickle ###############

save = True
if save:
  
  # save inside values: averages, full lists
  # and first 20 iterations as text
  values = {'yfs': yfs, 'ygs': ygs, 'yhs': yhs, 'yis': yis, 'yjs': yjs, 'yks': yks, 'yls': yls, 'avgf': avgf, 'avgg': avgg, 'avgh': avgh, 'avgi': avgi, 'avgj': avgj, 'avgk': avgk, 'avgl': avgl, "text": text}

  r = ""
  if nb_iter > 100:
    r = "".join([chr(random.randint(ord('A'), ord('Z'))) for i in range(5)])
  out_name = "experiments/{}.p".format(nameExp)
  
  with open(out_name, "wb") as f:
    pickle.dump( values, f )



print("Finished.")
