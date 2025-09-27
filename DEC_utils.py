import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def smoothNyquist(type,Xi,Yi,samplingRate,octaveWindowWidth,octaveWindowShift,xLimit):
    import math
    X  = []
    Y  = []
    #
    # shortest period (highest frequency) center of the window
    # starts  at the Nyquist
    #
    windowWidth = octaveWindowWidth
    halfWindow  = float(windowWidth / 2.0)
    windowShift = octaveWindowShift
    shift       = math.pow(2.0,windowShift)        # shift of each Fc

    #
    # the first center X at the Nyquist
    #
    if type == "frequency":
        Xc = float(samplingRate) / float(2.0) # Nyquist frequency
    else:
        Xc = float(2.0) /float(samplingRate)  # Nyquist period

    while (True):
        #
        # do not go below the minimum frequency
        # do not go above the maximum period
        #
        if (type == "frequency" and Xc < xLimit) or (type == "period" and Xc > xLimit):
            break

        thisBin = getBin(Xi,Yi,Xc,halfWindow)

        #
        # bin should not be empty
        #
        if (len(thisBin) > 0):
            Y.append(np.mean(thisBin))
            X.append(Xc)
        else:
            Y.append(float('NAN'))
            X.append(Xc)

        #
        # move the center frequency to the right by half of the windowWidth
        # move the center period to the left by half of the windowWidth
        #
        if type == "frequency":
            Xc /= shift
        else:
            Xc *= shift
    #
    # sort on X and return
    #
    X,Y = (list(t) for t in zip(*sorted(zip(X,Y))))
    return (X,Y)

def get_stft_freq():
    """ Get actual frequency labels for frequency samples """
    X = np.ones(60*250)
    # original frequency
    f, t, Zxx = signal.stft(X, fs=250, nperseg = 512)
    # frequency bins using 1/16 octave intervals
    subfreq = np.array(smoothNyquist('frequency',[],[],250,1/8,1/16,1))[0][:-1]
    subfreq = np.flip(subfreq)
    # Double check if there is frequency samples within each bin (xl - xu)
    shift = np.power(2, 1/32)
    freq = []
    for i in range(len(subfreq)):
        xl = subfreq[i]/shift
        xu = subfreq[i]*shift
        index = []
        for j in range(len(f)):
            if (f[j]>xl) & (f[j]<xu):
                index.append(j)
        if len(index)!=0:
            freq.append(subfreq[i])
    freq = np.array(freq)
    return freq

def getBin(X,Y,Xc,octaveHalfWindow):
    import math
    thisBin = []
    shift = math.pow(2.0,octaveHalfWindow)
    #
    # the bin is octaveHalfWindow around Xc
    #
    X1 = Xc / shift
    X2 = Xc * shift
    Xs = min(X1,X2)
    Xe = max(X1,X2)

    #
    # gather the values that fall within the range >=Xs and <= Xe
    #
    for i in range(0,len(X)):
        if X[i] >= Xs and X[i] <= Xe:
            thisBin.append(float(Y[i]))

    return thisBin

def switch_label(y_old, old):
    """
    Reorder the predicted labels: 
    Label 0: mainly high-frequency anthropogenic noise (> 20 Hz). [background noise]
    Label 1: discontinuous low-frequency rain-induced noise. [noise after the end of rain ]
    Label 2: continuous low-frequency rain-induced noise. [noise during moderate rain]
    Label 3: both high-frequency and low-frequency noise. [noise during heavy rain]
    y_old: (n_smaples)
    old: (n_clusters)
    """
    y_new = np.zeros_like(y_old)
    new = np.arange(0, len(old))
    for i in range(len(old)):
        y_new[y_old==old[i]] = new[i]
    return y_new

def plot_tsne():
    fig, ax = plt.subplots(figsize=(5,5))
    for i in range(args.n_clusters):
        class_idx = np.where(label==i)[0]
        ax.scatter(tsne_results[class_idx,0], tsne_results[class_idx,1], s=10, alpha=0.7, edgecolors='black', linewidths=0.5)

    ax.scatter(tsne_results[-center.shape[0]:,0], tsne_results[-center.shape[0]:,1], s=200, c='y' ,marker='*', edgecolors='black', linewidths=0.5)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()

def fft_taper(data):
    """
    Cosine taper, 10 percent at each end
    Inplace operation, so data should be float.
    """
    from obspy.signal.invsim import cosine_taper
    data *= cosine_taper(len(data), 0.2)
    return data

def cal_energy(xf, PSD, f1, f2):
    energy = np.sqrt(np.trapz(PSD[np.where((xf>=f1) & (xf<=f2))], xf[np.where((xf>=f1) & (xf<=f2))]))
    return energy