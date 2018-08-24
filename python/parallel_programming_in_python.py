import sys
import cPickle
import multiprocessing  
# from joblib import Parallel, delayed
import numpy as np



k = cPickle.load(open(sys.argv[1]))

list_j = [j for i in k[0] for j in i]


pool = multiprocessing.Pool(12)
output = pool.map(lambda x: x, list_j)
pool.close()
pool.join()


print("\n")
for k in output:
    print k.shape

