import pandas as pd
import numpy as np
import neurokit as nk
import pysiology as ps
from pysiology.electrodermalactivity import phasicGSRFilter


def filterGSR(rawGSRSignal,samplerate,seconds,scr_treshold):
    fgsr = nk.eda_process(rawGSRSignal,samplerate,scr_treshold=scr_treshold)
    return fgsr

def tonicGSRFilter(rawGSRSignal, samplerate, seconds=4):
    """ Apply a modified filter to the signal, with +- X seconds from each sample, in order to extract the tonic component. Default is 4 seconds

        * Input:
            * rawGSRSignal = gsr signal as list
            * samplerate = samplerate of the signal
            * seconds = number of seconds before and after each timepoint to use in order to compute the filtered value
        * Output:
            * tonic signal

        :param rawGSRSignal: raw GSR Signal
        :type rawGSRSignal: list
        :param samplerate: samplerate of the GSR signal in Hz
        :type samplerate: int
        :param seconds: seconds to use to apply the phasic filter
        :param seconds: int
        :return: filtered signal
        :rtype: list

    """

    tonicSignal = []
    for sample in range(0, len(rawGSRSignal)):
        smin = sample - seconds * samplerate  # min sample index
        smax = sample + seconds * samplerate  # max sample index
        # is smin is < 0 or smax > signal length, fix it to the closest real sample
        if (smin < 0):
            smin = sample
        if (smax > len(rawGSRSignal)):
            smax = sample
        # substract the mean of the segment
        newsample = np.mean(rawGSRSignal[smin:smax])
        # move to th
        tonicSignal.append(newsample)
    return (tonicSignal)


def phasicGSRFilter(rawGSRSignal, samplerate, seconds=4):
    """ Apply a phasic filter to the signal, with +- X seconds from each sample. Default is 4 seconds

        * Input:
            * rawGSRSignal = gsr signal as list
            * samplerate = samplerate of the signal
            * seconds = number of seconds before and after each timepoint to use in order to compute the filtered value
        * Output:
            * phasic signal

        :param rawGSRSignal: raw GSR Signal
        :type rawGSRSignal: list
        :param samplerate: samplerate of the GSR signal in Hz
        :type samplerate: int
        :param seconds: seconds to use to apply the phasic filter
        :param seconds: int
        :return: filtered signal
        :rtype: list

    """

    phasicSignal = []
    for sample in range(0, len(rawGSRSignal)):
        smin = sample - seconds * samplerate  # min sample index
        smax = sample + seconds * samplerate  # max sample index
        # is smin is < 0 or smax > signal length, fix it to the closest real sample
        if (smin < 0):
            smin = sample
        if (smax > len(rawGSRSignal)):
            smax = sample
        # substract the mean of the segment
        newsample = rawGSRSignal[sample] - np.mean(rawGSRSignal[smin:smax])

        # move to th
        phasicSignal.append(newsample)
    return (phasicSignal)
