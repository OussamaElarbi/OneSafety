import math
import pandas as pd
# dataRead() verifies the input data for error free
try:
	df=pd.read_csv('Data/GSR_HR_DATA.csv',sep=',')
	col = len(df.columns);
	index = len(df.index);
	hr=pd.DataFrame()
	gsr=pd.DataFrame()
	time=pd.DataFrame()
	for c in df.columns:
		if(c=='Time'):
			time= df['Time']
		elif(c=='HR'):
			hr=df['HR']
		elif(c=='GSR'):
			gsr=df['GSR']
	if(hr.empty|gsr.empty|time.empty):
		print('Missing data! Please check HR,GSR and TimeStamp exist')
		exit(1)
	#Round dataframe values to 2 decimls after '.'
	df=df.round(2)
	# Clean the data from missing values
	for row in df.head(index).itertuples():
		if((row.Time == '-1')|(pd.isnull(row.Time))|(row.Time == '0')):
			df = df.drop(row.Index)
		elif ((row.GSR == -1) | (math.isnan(row.GSR)) | (row.GSR == 0)):
			df = df.drop(row.Index)
		elif ((row.HR == -1) | (row.HR == 0) | (math.isnan(row.HR))):
			df = df.drop(row.Index)
	#Update HR,GSR & TIME Datafarme seperaltly
	hr = df['HR']
	gsr = df['GSR']
	time= df['Time']

except pd.errors.EmptyDataError :
		print("No columns to parse from file,please re-upload again!")