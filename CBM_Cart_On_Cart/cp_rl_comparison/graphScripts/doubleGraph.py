#!/usr/bin/python
import argparse
from common import *

parser = argparse.ArgumentParser(description='Creates a double graph of performance. Provide logfiles for both the RL agent as well as the Cerebellum agent.')
parser.add_argument('-r', metavar='r', type=str, nargs='+',
                   help='Logfiles for RL agent')
parser.add_argument('-c', metavar='c', type=str, nargs='+',
                   help='Logfiles for Cerebellum agent')

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

from pylab import figure, show
rl_aloft, rl_dev = proccessLogs(args.r)
cb_aloft, cb_dev = proccessLogs(args.c)
x = range(1,len(rl_aloft)+1)
fig = figure(1)

ax1 = fig.add_subplot(211)
p1 = ax1.plot(x, rl_aloft,'o--')
p2 = ax1.plot(x, cb_aloft,'*-')
#ax1.grid(True)
ax1.set_ylim((-5,105))
ax1.set_ylabel('Success Percentage')
#ax1.set_title('')

# for label in ax1.get_xticklabels():
#     label.set_color('r')
ax1.legend((p1[0], p2[0]),('Q-Learning', 'Cerebellum'), loc=4)

ax2 = fig.add_subplot(212)
ax2.plot(x, rl_dev,'o--')
ax2.plot(x, cb_dev,'*-')
#ax2.grid(True)
ax2.set_ylabel('Avg Angular Deviation')
#ax2.set_ylim( (-2,2) )
l = ax2.set_xlabel('Trial Number')
#l.set_color('g')
#l.set_fontsize('large')

show()
