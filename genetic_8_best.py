from scipy.stats import zscore 
import numpy as np
import pandas as pd
from operator import itemgetter
from multiprocessing import Pool
from copy import deepcopy
from time import time
import itertools, sys, csv, multiprocessing, random, math

'''
MATCHED GROUPS ALGORITHM!

usage:
python3 /path/to/this/file /path/to/input/file <integer>
   – where the integer indicates the number of groups to be matched
   - outputs results to current working directory.

Current assumptions:
   - each variable is equally important (no weighting is performed).
   - every case has the same number of variables as the first (NON-JAGGED ARRAYS)
   - all variables have been z-scored.

goal:
   - get average scores as close to zero as possible across variables in each goup.
   - avoid differences in variance across groups.

future optimization goals:
   - use numpy arrays in lieu of 2d arrays when possible to avoid highly repeated conversions
'''

def deviancesGroups(l3d):
   '''Assumes equal variance and means across all variables. z-score your values before putting them in this algorithm.'''
   flat = flatten(l3d)
   num_vars = len(l3d[0][0])
   group_lens = [len(l2d) for l2d in l3d]
   mean, dev = 0,1 
   mean_devs = []
   dev_devs = []
   cols = []
   i = 0
   for l in group_lens:
      group = flat[i:i + l]
      for coli in range(num_vars):
         col = [group[coli*period] for period in range(len(group) // num_vars)]
         mean_devs.append(abs(mean - np.mean(col)))
         dev_devs.append(abs(dev - np.std(col)))
      i = i + l
   out = ([np.mean(mean_devs[i:i+num_vars]) for i in range(0, len(mean_devs), num_vars)], [np.mean(dev_devs[i:i+num_vars]) for i in range(0, len(dev_devs), num_vars)])
   assert (len(out[0]) == len(out[1])) and len(out[0]) == len(l3d)
   return out #returns the average deviances for each group in the format ([group mean dev1, group mean dev2, etc.], [group dev dev1, group dev dev2, etc.])

def flatten(l3d):
   try:
      return [v for l2d in l3d for l1d in l2d for v in l1d]
   except TypeError: #expectency: l1d is float, not a list.
      return [l1d for l2d in l3d for l1d in l2d]

def sort4d(l4d):
   scores = []
   for l3d in l4d:
      s = deviancesGroups(l3d)
      scores.append(sum([sum([s[0][i], s[1][i]]) for i in range(len(s[0]))])) #assumes even weighting of mean deviances and deviance deviances. Since there is a constant scale and number of vars for each data point, we don't need to average or otherwise standardize.
   return reorderList(l4d, np.argsort(scores))

def reorderList(original, newOrder): #should be rewritten to work in place to avoid rewriting to memory
   '''newOrder is the list of the original indices of the list to be ordered, given in the order they should appear in the reordered version. Does not deep copy.'''
   assert len(original) == len(newOrder)
   ordered = []
   for i in newOrder:
      ordered.append(original[i])
   return ordered

def importFromCSV(filename):
   p = pd.read_csv(filename)
   return p[['PSEGCustomers','Age','MedianRoomNumber','percentOwnerOccupied','PoliticalOrientationpercentdemin2012','Total','Population','Housingunits','PSEGHHs','Income','Highschoolormore','Total2','Total3','Total4','Population2','Population3']], p 

def generation(l3d):
   l3d_c = deepcopy(l3d)
   for l2d in l3d_c:
      random.shuffle(l2d)
   return (reproduce(l3d_c))

def reproduce(l3d):
   '''PARAMETRICITY: RELIES ON MEANS AND STANDARD DEVIATIONS. takes in a sorted 3D list where each 2D list is a current group. Has them reproduce.'''
   mean, dev = 0, 1 #get_grands(flatten(l3d)) #WILL NEED TO BE CHANGED FROM NUMBERS TO LISTS OF NUMBERS IF MEANS OR DEVIANCES ARE EQUAL ACROSS VARIABLES.
   new = []
   random.shuffle(l3d)
   for x in range((len(l3d))//2):
      new.extend(idealHalves(l3d[x*2],l3d[x*2+1], [mean] * len(l3d[0][0]), [dev] * len(l3d[0][0]))) #ASSUMES MEANS + DEVIANCES ARE EQUAL ACROSS VARIABLES.
   if len(l3d) % 2 != 0:
      new.append(l3d[-1])
   assert len(l3d) == len(new)
   return new

def idealHalves(l2d1, l2d2, ms, ds):
   '''PARAMETRICITY: RELIES ON MEANS AND STANDARD DEVIATIONS. finds the ideal recombination of halves so that one minimizes deviance and the other preserves the remaining cases.'''
   ar1 = np.array(l2d1)
   ar2 = np.array(l2d2)
   halves = [ar1[:len(ar1)//2], ar1[len(ar1)//2:], ar2[:len(ar2)//2], ar2[len(ar2)//2:]]
   possible = []
   devs = []
   i = 0 
   for x in range(len(halves)):
      for y in range(x+1,len(halves)):
         possible.append(np.array([l1d for l2d in [halves[x],halves[y]] for l1d in l2d]))
   assert len(possible) == 6
   for p in possible:
       devs.append(get_dev(p, ms, ds))
   complements = {0:5, 1:4, 2:3, 3:2, 4:1, 5:0} #the indices of each permutation in possible and its complement
   i = devs.index(min(devs))
   out1 = possible[i].tolist()
   out2 = possible[complements[i]].tolist()
   assert sum([len(out1),len(out2)]) == sum([len(l2d1),len(l2d2)])
   return [out1,out2]

def get_dev(p, ms, ds):
    '''p is a 2d array. this function returns the sum of the absolute deviation from the mean and the absolute difference of the standard deviation from the mean and this deviation, averaged across all variables.'''
    m = np.mean([abs(ms[i] - np.mean(p[:,i])) for i in range(p.shape[1])]) #difference of mean of this group from the grand mean
    d = np.mean([abs(ds[i] - abs(ms[i] - np.mean(p[:,i]))) for i in range(p.shape[1])]) #difference of the deviation of this group from the total deviance
    return m + d

def n_generations(l3d, n):
   len_list = len([i for l in l3d for i in l])
   avg_devs = []
   while n > 0:
      devs = [v for g in deviancesGroups(l3d) for v in g]
      avg_devs.append([sum(devs)/len(devs)])
      new_gen = generation(l3d)
      l3d = sort4d([l3d,new_gen])[0] #if offspring are worse, revert to parents and try again.
      n = n - 1
      fl = [i for l in new_gen for i in l]
      assert len(fl) == len_list #total number of features are preserved
   with open('ngen_deviance_curve_over_iters_'+str(time()).replace(".","_")+'.csv', "w") as f:
      writer = csv.writer(f)
      writer.writerows(avg_devs)
   return l3d

def firstGroups(cases, number_groups, versions):
   l4d = []
   for v in range(versions):
      c_perm = np.random.permutation(cases)
      l4d.append(g(c_perm.tolist(), number_groups))
   return l4d

def g(c_perm, numbergroups):
   l3d = []
   tot = len(c_perm)
   pergroup = tot // numbergroups
   mod = tot % numbergroups
   for n in range(numbergroups):
      l3d.append(c_perm[n*pergroup:(n*pergroup) + pergroup])
   for n in range(mod):
      l3d[n].append(c_perm[(numbergroups*pergroup) + n])
   print(l3d)
   return l3d

if __name__ == '__main__':
   vs = ['PSEGCustomers','Age','MedianRoomNumber','percentOwnerOccupied','PoliticalOrientationpercentdemin2012','Total','Population','Housingunits','PSEGHHs','Income','Highschoolormore','Total2','Total3','Total4','Population2','Population3']
   
   allcases, df = importFromCSV(sys.argv[1]) #read in CSV (for formatting, see sample input file)
   print("imported data. Number of cases: "+str(len(allcases)))
   number_groups = int(sys.argv[2]) #number of groups to produce
   
   generations = 10 #number of generations per population
   num_pops = 1 #number of populations that are independently reproducing for <generations> number of generations
   
   np.random.seed(seed=478) #seed set for reproducibility
   random.seed(a=478)
   
   # generate populations
   print("permutating data to generate populations.")
   l4d = firstGroups(allcases, number_groups, num_pops)
   print(str(num_pops) + " populations generated with "+ str(number_groups)+" groups each. Beginning multiprocessing for simultaneous reproduction across populations.")
   
   # reproduce. Since each population is independent, each can run as a distinct process.
   num_cores = multiprocessing.cpu_count()
   with Pool(num_cores) as p:
      l4d = p.starmap(n_generations, [(l3d, generations) for l3d in l4d])
   
   # find the best-matched set out of the all of the populations
   bests = sort4d(l4d)[0:8]
   for i in range(len(bests)):
      best = bests[i] 
      condition = 0
      a = pd.DataFrame()
      for l2d in best:
         t = pd.DataFrame(l2d)
         t.columns = ['PSEGCustomers','Age','MedianRoomNumber','percentOwnerOccupied','PoliticalOrientationpercentdemin2012','Total','Population','Housingunits','PSEGHHs','Income','Highschoolormore','Total2','Total3','Total4','Population2','Population3']
         t['Condition'] = condition
         a = a.append(t)
         condition = condition + 1
      d = df.merge(a)
      d.to_csv("genetic_output_"+str(i)+sys.argv[1].split("/")[-1])

   
