import hrvanalysis as hrv


# Features Extractions for Heart Rate Variability
# Time domain features : Mean_NNI, SDNN, SDSD, RMSSD, Median_NN, Range_NN, CVSD, CV_NNI, Mean_HR, Max_HR, Min_HR, STD_HR

def time_domain_features(rr):
    time_features = hrv.get_time_domain_features(rr).values()
    return time_features


# Frequency domain features : LF, HF, VLF, LH/HF ratio, LFnu, HFnu, Total_Power

def frequency_domain_features(rr):
    frequency_features = hrv.get_frequency_domain_features(rr).values()
    return frequency_features


# poincare domain features :
def poincare_domain_features(rr):
    features = hrv.extract_features.get_poincare_plot_features(rr).values()
    return features


def psd_plot(rr):
    plot = hrv.plot_psd(rr, method="welch")
    return plot


def dist_plot(rr):
    plot = hrv.plot_distrib(rr, method="lomb")
    return plot
