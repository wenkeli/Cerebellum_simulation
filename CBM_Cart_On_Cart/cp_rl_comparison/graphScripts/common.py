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
