import math
from builtins import KeyError, AttributeError, set
import hrvanalysis as hrv
import pandas as pd
import tabulate as tb
from collections import Counter

import numpy as np
# dataRead() verifies the input data for error free

try:
	df=pd.read_csv('Data/GSR_HR_DATA.csv',sep=',')
	col = len(df.columns);
	index = len(df.index);
	#Round dataframe values to 2 decimls after '.'
	df=df.round(2)
	# Clean the data from Nan ,missing and duplicate values
	df=df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
	for row in df.head(index).itertuples():
		if((row.Time == '-1')|(row.Time == '0')):
			df = df.drop(row.Index)
		elif ((row.GSR == -1) | (row.GSR == 0)):
			df = df.drop(row.Index)
		elif ((row.HR == -1) | (row.HR == 0)):
			df = df.drop(row.Index)
	# Select duplicate rows except first occurrence based on all columns
	duplicateRowsDF = df[df.duplicated(['Time'])]
	for i in duplicateRowsDF.itertuples():
		df = df.drop(i[0])

except pd.errors.EmptyDataError:
	print("No columns to parse from file,please re-upload again!")
except KeyError:
	raise KeyError('Missing data! Please check HR,GSR and TimeStamp exist')
except AttributeError:
	print('Missing data! Please check HR,GSR and TimeStamp data exist')

#Data Preprocessing
# Extracting the RR intervals HR(BPM) ==> HRV(ms): 1bpm =60,000 ms
rr = []
for row in df.head(index).itertuples():
	rr.append(int((60000 / row.HR)))
df['RR'] = rr

# This remove outliers from signal
rr_without_outliers = hrv.remove_outliers(rr_intervals=rr,  low_rri=300, high_rri=2000)
# This replace outliers nan values with linear interpolation
interpolated_rr_intervals = hrv.interpolate_nan_values(rr_intervals=rr_without_outliers, interpolation_method="linear")
df['RR_Interpolated'] = interpolated_rr_intervals
# This remove ectopic beats from signal
nn_intervals_list = hrv.remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="malik")
# This replace ectopic beats nan values with linear interpolation
interpolated_nn_intervals = hrv.interpolate_nan_values(rr_intervals=nn_intervals_list)
df['NN_Interpolated'] = interpolated_nn_intervals
# Round dataframe values to 2 decimls after '.'
df = df.round(2)
#df['RR'].to_csv(path_or_buf='/Users/smileyboy/PycharmProjects/OneSafety/Data/RR.csv', sep=',')


#Features Extractions for Heart Rate
#Time domain features : Mean_NNI, SDNN, SDSD, RMSSD, Median_NN, Range_NN, CVSD, CV_NNI, Mean_HR, Max_HR, Min_HR, STD_HR
time_domain_features = hrv.get_time_domain_features(nn_intervals_list)
print(time_domain_features)

#Frequency domain features : LF, HF, VLF, LH/HF ratio, LFnu, HFnu, Total_Power
frequency_domain_features= hrv.get_frequency_domain_features(nn_intervals_list,'welch',128, 'linear')
print(frequency_domain_features)

#Features Extractions for GSR