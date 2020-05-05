#!/usr/bin/python3

import itertools
from numpy.random import dirichlet
import sys
import matplotlib.pyplot as plt
import pickle
import random

"""
this file contains some utility functions that will be used by our experiment scripts
"""

# cartesian product
iproduct = itertools.product

def avg(l):
  if l==[]: return 0
  return sum(l)/len(l)

def pprint(x):
  print(x, flush=True, end='')

############### generate randomisations ###############

# additive approximation
def gen_g(f):
  def g(*xs):
    return f(*xs[:-1]) + xs[-1]
  return g

# truncation
def gen_h(f, d):
  def h(*xs):
    o = f(*xs)
    return o - (o % (2*d + 1)) + d
  return h

############### input distributions ###############

# builds a uniform distribution
def uni(domain):
  return {k:1/len(domain) for k in domain}

# triangular distribution
def tri(domain):
  n = len(domain)
  return {k:(i+1)*2/(n*(n+1)) for i,k in enumerate(domain)}

# "decreasing" triangular distribution
def tri_down(domain):
  n = len(domain)
  return {k:(n-i)*2/(n*(n+1)) for i,k in enumerate(domain)}

## noise distributions below

# defines random distribution on domain, drawn from Dirichlet symmetric(1)
p_dirichlet = 1
def r_dist(domain):
  d = dirichlet([p_dirichlet]*len(domain))
  return {x: d[i] for i,x in enumerate(domain)}

# truncated symmetric geometric distribution around 0, max distortion delta, parameter p in [0,1]
def trunc_sym_geom(p, delta):
  h = {k: p**abs(k)*(1-p)/(1+p) for k in range(-delta, delta+1)}
  s = sum(h.values())
  return {x: h[x]/s for x in h}


############### parse arguments ###############

def parseArgs(maxi_default, nb_iter_default, save=True):
  args = sys.argv
  if len(args) == 1:
    # default parameters
    maxi = maxi_default
    nb_iter = nb_iter_default
  else:
    try:
      assert(len(args) == 3 or len(args) == 4)
      maxi = int(args[1])
      nb_iter = int(args[2])
      assert(maxi > 0 and nb_iter > 0)
      if len(args) == 4:
        assert(args[3] == 'nosave')
        save = False
    except: 
      print("Please use as follows: \n- either provide no argument\n- or provide 2 positive integers corresponding to the maximal input size and the number of iterations respectively (optionally followed with the flag 'nosave'). ")
      sys.exit(42)
  
  print("Starting analysis with the following parameters:")
  print("   Maximal input value  = {}".format(maxi))
  print("   Number of iterations = {}".format(nb_iter))
  if not save:
    print("Results will not be saved. ")
  return (maxi, nb_iter, save)


############### analyses ###############

# labels

lf = "Original function"
lg = "Uniform randomisation"
lh = "Truncated randomisation"
li = "Independently optimal"
lj = "Deterministic merge"
lk = "Dynamic merge"
ll = "Truncated Laplace"


def compute_averages(yfs, ygs, yhs, yis, yjs, yks, yls):
  withF = yfs != None
  if withF: 
    avgf = avg(yfs)
  else:
    avgf = None
  avgg = avg(ygs)
  avgh = avg(yhs)
  avgi = avg(yis)
  avgj = avg(yjs)
  avgk = avg(yks)
  avgl = avg(yls)

  print("\n# Average values:")
  if withF: print(lf + " =       {}".format(avgf))
  print(lg + " =   {}".format(avgg))
  print(lh + " = {}".format(avgh))
  print(li + " =   {}".format(avgi))
  print(lj + " =     {}".format(avgj))
  print(lk + " =           {}".format(avgk))
  print(ll + " =       {}".format(avgl))
  
  return avgf, avgg, avgh, avgi, avgj, avgk, avgl
  


nFirst = 20

def plot_values(xfs, yfs, ygs, yhs, yis, yjs, yks, yls, optimal, nameExp, tag, utility=False):
  withF = yfs != None
  if withF: plt.plot(xfs[:nFirst], yfs[:nFirst], 'bx-', label=lf)
  plt.plot(xfs[:nFirst], ygs[:nFirst], 'gx:', label=lg)
  plt.plot(xfs[:nFirst], yhs[:nFirst], 'cx-.', label=lh)
  plt.plot(xfs[:nFirst], yjs[:nFirst], 'kx--', label=lj)
  plt.plot(xfs[:nFirst], yks[:nFirst], 'rx:', label=lk)
  plt.plot(xfs[:nFirst], yls[:nFirst], 'yx:', label=ll)
  if optimal:
    plt.plot(xfs[:nFirst], yis[:nFirst], 'mx-', label=li)

  title = tag
  plt.title(title)
  plt.xlabel("iteration")
  plt.ylabel(tag)
  plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  
  
  outDir = "experimentsUtility" if utility else "experiments"
  plt.savefig(outDir + "/{}.pdf".format(nameExp), bbox_inches='tight')
  print("   Data plotted in {}".format(outDir + "/{}.pdf".format(nameExp)))


def analyse(xfs, yfs, ygs, yhs, yis, yjs, yks, yls, optimal, nameExp, save=True, utility=False):

  withF = yfs != None
  avgf, avgg, avgh, avgi, avgj, avgk, avgl = compute_averages(yfs, ygs, yhs, yis, yjs, yks, yls)

  text = ""
  
  if withF:
    if optimal:
      text += ("x fx uni opt trunc det dyn lap\n")
    else:
      text += ("x fx uni trunc det dyn lap\n")
  else:
    if optimal:
      text += ("x uni opt trunc det dyn lap\n")
    else:
      text += ("x uni trunc det dyn lap\n")

  if withF:

    for i in range(min(len(ygs), nFirst)):
      if optimal:
        text += ("{} {} {} {} {} {} {} {}\n".format(i, yfs[i], ygs[i], yis[i], yhs[i], yjs[i], yks[i], yls[i]))
      else:
        text += ("{} {} {} {} {} {} {}\n".format(i, yfs[i], ygs[i], yhs[i], yjs[i], yks[i], yls[i]))
  
  else:
    for i in range(min(len(ygs), nFirst)):
      if optimal:
        text += ("{} {} {} {} {} {} {}\n".format(i, ygs[i], yis[i], yhs[i], yjs[i], yks[i], yls[i]))
      else:
        text += ("{} {} {} {} {} {}\n".format(i, ygs[i], yhs[i], yjs[i], yks[i], yls[i]))
  
  print("\n# First values:")
  print(text)

  ############### pickle ###############

  if save:
    print("Saving data ...")

    if withF:
      values = {'yfs': yfs, 'ygs': ygs, 'yhs': yhs, 'yis': yis, 'yjs': yjs, 'yks': yks, 'yls': yls, 'avgf': avgf, 'avgg': avgg, 'avgh': avgh, 'avgi': avgi, 'avgj': avgj, 'avgk': avgk, 'avgl': avgl, "text": text}

    else:
      values = {'ygs': ygs, 'yhs': yhs, 'yis': yis, 'yjs': yjs, 'yks': yks, 'yls': yls, 'avgg': avgg, 'avgh': avgh, 'avgi': avgi, 'avgj': avgj, 'avgk': avgk, 'avgl': avgl, "text": text}

    r = ""
    if len(ygs) > 100:
      r = "_" + "".join([chr(random.randint(ord('A'), ord('Z'))) for i in range(5)])
    
    outDir = "experimentsUtility" if utility else "experiments"
    out_name = outDir + "/{}.p".format(nameExp + r)
    
    with open(out_name, "wb") as f:
      pickle.dump( values, f )
    
    print("   Data saved in {}".format(out_name))

