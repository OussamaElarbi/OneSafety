import pysiology as ps
import pandas as pd

df=pd.read_csv('Data/GSR_HR_DATA.csv',sep=',')
df= df.iloc[15000:]
df=df.round(decimals=2)
gsr=df['GSR'].tolist()
print(gsr)
peak= ps.electrodermalactivity.findPeakOnsetAndOffset(rawGSRSignal=gsr)
print(peak)
gsr_processed= ps.electrodermalactivity.GSRSCRFeaturesExtraction(filteredGSRSignal=gsr,samplerate=4,peak=peak)
print(gsr_processed)
ph_to= ps.electrodermalactivity.getPhasicAndTonic(gsr,4,4)
print(ph_to)
ph= ps.electrodermalactivity.phasicGSRFilter(ph_to,4,4)
print(ph)

