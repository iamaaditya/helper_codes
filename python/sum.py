#!/usr/bin/python

import sys

val = map(float, open(sys.argv[1]).read().splitlines())
print "sum: ", sum(val)
print "avg: ", sum(val)/len(val)

