import biosppy
import neurokit as nk
import pandas as pd
from matplotlib import pyplot as plt
import data_pre_processing as dpp
import hrv_analysis as hrv
import gsr_analysis as gsra
import scipy as sp
import statistics as ss
'''
Define HRV Variables
'''
mean_rr = sd_rr = sdsd = rmssd = med_rr = mean_hr = max_hr = min_hr = sd_hr = lf = hf = lf_hf = lfnu = hfnu = total_power = vlf = 0
'''
Define GSR Variables
'''
mean_gsr = med_gsr = sd_gsr = 0
'''
Define Accelerometer Variables
'''
x = y = z = 0

file = '/Users/smileyboy/OneSafety/Data/GSR_HR_DATA.csv'
df = dpp.data_read(file)
gsr=df['GSR']
gsr =gsra.filterGSR(gsr,samplerate=16,seconds=2,scr_treshold=100)
rgsr = gsr['df']['EDA_Raw'].tolist()
fgsr = gsr['df']['EDA_Filtered'].tolist()
peaks= sp.signal.find_peaks(fgsr)
print(peaks)
print(ss.mean(fgsr),ss.median(fgsr))
phasic = gsra.phasicGSRFilter(fgsr,samplerate=16,seconds=2)
tonic = gsra.tonicGSRFilter(fgsr,samplerate=16,seconds=2)
#phasic = fgsr[0:50]+fgsr[75:85]+fgsr[85:-1]
plt.title('Raw/Filtered GSR')
plt.xlabel('Time(s)')
plt.ylabel('Skin Conductance')
plt.plot(rgsr,label='Raw GSR')
plt.plot(fgsr,label='Filtered GSR')
plt.legend()
plt.show()
plt.plot(phasic,label='Phasic signal')
plt.xlabel('Time(s)')
plt.ylabel('Skin Conductance')
plt.legend()
plt.show()
plt.plot(tonic,label='Tonic signal')
plt.xlabel('Time(s)')
plt.ylabel('Skin Conductance')
plt.legend()
plt.show()

file = '/Users/smileyboy/OneSafety/Data/GSR_HR_DATA.csv'
# Process the shimmer GSR+ data
df = dpp.data_read(file)
hr = df['HR']
hr = hr.tolist()
# Obtain RR interval list
rr = dpp.rr_process(hr)
# Conduct HRV time analysis
mean_rr , sdnn , sdsd , nni_50,pnni_50,nni_20,pnni_20,rmssd , mednn ,range_rr,cvsd,cvnni, mean_hr ,max_hr ,min_hr ,sd_hr =hrv.time_domain_features(rr)
# Conduct HRV frequency analysis
lf, hf, lf_hf, lfnu, hfnu, total_power, vlf = hrv.frequency_domain_features(rr)
# Conduct HRV poincare analysis
sd1, sd2, sd1sd2 = hrv.poincare_domain_features(rr)
# plot hrv data
print(range_rr,rmssd)
hrv.hrv.plot.plot_timeseries(rr)
hrv.hrv.plot.plot_distrib(rr)
hrv.hrv.plot.plot_poincare(rr)
hrv.hrv.plot.plot_psd(rr)

'''
p_gsr=nk.eda_process(gsr,sampling_rate=128,scr_treshold=0.1)
list_key_value = [ [k,v] for k, v in p_gsr.items() ]
df=p_gsr['df']
df=df.round(2)
df.to_csv('EDA_Filtered.csv',sep=',')
eda = df['EDA_Raw']
eda = eda.tolist()
feda = df['EDA_Filtered']
feda = feda.tolist()
df['index']=df.index
'''

# gsr = df['GSR'].tolist()
# Features Extractions for GSR
# pt.plot(df['Time'], gsr)
# pt.show()
