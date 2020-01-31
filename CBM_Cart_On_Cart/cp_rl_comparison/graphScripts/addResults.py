#!/usr/bin/python
import argparse
from common import *

parser = argparse.ArgumentParser(description='Reports the success percentage and average angular deviation of the first trials.')
parser.add_argument('-l', metavar='l', type=str, nargs='+',
                   help='Logfiles for agent')

args = parser.parse_args()

# Process the logfiles
def proccessLogs(logs):
    aggAloft = []
    aggDev = []
    for log in logs:
        aloft = balanced(log)
        avgAbsDev = getAvgAbsAngDev(log)
        assert len(aloft) == len(avgAbsDev)
        aggAloft.append(aloft)
        aggDev.append(avgAbsDev)
    aggAloftSum = [sum(a) for a in zip(*aggAloft)]
    aggDevSum = [sum(a) for a in zip(*aggDev)]
    avgAggAloft = [i/float(len(logs)) for i in aggAloftSum]
    avgAggDev = [i/float(len(logs)) for i in aggDevSum]        
    return avgAggAloft, avgAggDev

aloft, dev = proccessLogs(args.l)
print 'Success Perc Raw: ', aloft
print 'Success Percentages: ',
for i in aloft:
    print int(i), ' & ',
print
print 'Ang Dev Raw: ', dev
print 'Angular Deviations: ',
for i in dev:
    print "%.2f" % i, ' & ',
