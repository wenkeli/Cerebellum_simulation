#!/usr/bin/python
import sys
import getopt
import matplotlib.pyplot as plt

# How long is the pole aloft in each trial?
def getAloftTimes(logfile):
    aloft = []
    f = open(logfile)
    for line in f:
        if 'TimeAloft' in line:
            s = line.split()
            aloft.append(int(s[2]))
    f.close()
    return aloft

# Was the pole balanced or not for each trial?
def balanced(logfile):
    aloft = getAloftTimes(logfile)
    # Here we assume max aloft time is 10,000 timesteps
    return [100 if i >= 10000 else 0 for i in aloft]

# Returns a list of lists whichs contains the angular
# deviation at each interval for each trial
def getAngularDev(logfile):
    result = []
    angDev = []
    f = open(logfile)
    for line in f:
        if 'AbsTheta' in line:
            s = line.split()
            ang = float(s[2])
            angDev.append(ang)
        elif 'TimeAloft' in line:
            result.append(angDev)
            angDev = []
    f.close()
    return result

# Returns a list of the average absolute angular deviations
# for each trial.
def getAvgAbsAngDev(logfile):
    result = []
    devs = getAngularDev(logfile)
    for l in devs:
        absDev = [abs(i) for i in l]
        avgAbsDev = sum(absDev) / float(len(absDev))
        result.append(avgAbsDev)
    return result

# Plot how long the pole was aloft for each trial
def plotAloft(logfile):
    aloft = getAloftTimes(logfile[0])
    plt.plot(aloft,'b.')
    plt.ylabel('Time Aloft')
    plt.xlabel('Trial Number')
    plt.title('Aloft Time')
    plt.show()

# Angular deviations overliad upon each other
def plotAngOverlay(logfile):
    cycles, thetaError = [],[]
    f = open(logfile[0])
    start = None
    for line in f:
        if 'AbsTheta' in line:
            s = line.split()
            if not start:
                start = int(s[0])
            cycles.append(int(s[0]) - start)
            thetaError.append(float(s[2]))
        elif 'TimeAloft' in line:
            plt.plot(cycles,thetaError)
            start = None
            cycles,thetaError = [], []
    f.close()

    plt.ylabel('Angular Deviation (Degrees)')
    plt.xlabel('Milliseconds')
    plt.title('Angular Deviation')
    plt.show()

# Plots the angle of the pole as time progresses
def plotAng(logfile):
    cycles, thetaError = [],[]
    f = open(logfile[0])
    start = None
    for line in f:
        if 'AbsTheta' in line:
            s = line.split()
            if not start:
                start = int(s[0])
            cycles.append(s[0])
            thetaError.append(float(s[2]))
        elif 'TimeAloft' in line:
            plt.plot(cycles,thetaError,linewidth=2.0)
            start = None
            cycles,thetaError = [], []
    f.close()

    plt.ylabel('Angular Deviation (Degrees)')
    plt.xlabel('Milliseconds')
    plt.title('Angular Deviation')
    plt.show()
    
# Plots point version of angular deviations
def plotAbsAngCondensed(logfile):
    thetaError = []
    avgErrors = []
    f = open(logfile[0])
    for line in f:
        if 'AbsTheta' in line:
            s = line.split()
            ang = abs(float(s[2]))
            thetaError.append(ang)
        elif 'TimeAloft' in line:
            avgError = sum(thetaError) / float(len(thetaError))
            avgErrors.append(avgError)
            thetaError = []
    f.close()

    plt.plot(avgErrors,'b.')
    plt.ylabel('Mean Absolute Angluar Deviation (Degrees)')
    plt.xlabel('Trial Num')
    plt.title('Angular Deviation')
    plt.show()
    
# How probable was error to be delivered at each timestep
def plotErrProb(logfile):
    cycles, errorProb = [],[]
    f = open(logfile[0])
    for line in f:
        if 'ErrorProb' in line:
            s = line.split()
            cycles.append(int(s[0]))
            errorProb.append(float(s[2]))
    f.close()
    plt.plot(cycles,errorProb)
    plt.ylabel('Error Probability')
    plt.xlabel('Milliseconds')
    plt.title('Probability of Error Delivery')
    plt.show()

def test(arg):
    print 'given args: ', arg

def main():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--function", dest="function", help="This is the function that is eval'ed and passed the given arguments.")
    (options, args) = parser.parse_args()
    eval(options.function+'('+str(args)+')')

if __name__ == "__main__":
    main()
