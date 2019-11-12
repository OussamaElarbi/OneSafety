import time

import data_processing as dp

from sklearn import preprocessing

# The number of steps within one time segment
TIME_PERIODS = 80
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 40
# Load data set containing all the data from csv
df = dp.read_data('WISDM_ar_v1.1_raw.csv')
dp.show_basic_dataframe_info(df)
df.dropna(axis=0, how='any', inplace=True)
df['x-axis'] = dp.feature_normalize(df['x-axis'])
df['y-axis'] = dp.feature_normalize(df['y-axis'])
df['z-axis'] = dp.feature_normalize(df['z-axis'])

df=df.drop(columns='user-id')
df=df.replace('Walking','A')
df=df.replace('Jogging','B')
df=df.replace('Sitting','C')
df=df.replace('Standing','D')
df=df.replace('Downstairs','E')
df=df.replace('Upstairs','F')
df = df.rename(columns={"timestamp": "time"})
df = df.rename(columns={"x-axis": "x"})
df = df.rename(columns={"y-axis": "y"})
df = df.rename(columns={"z-axis": "z"})
df = df.round({'x': 4, 'y': 4, 'z': 4})
df.to_csv("output_filename.csv", index=False, encoding='utf8',sep=',')
print(df)
'''
for activity in np.unique(df['activity']):
    subset = df[df['activity'] == activity][:180]
    dp.plot_activity(activity, subset)
'''
# Define column name of the label vector
LABEL = 'ActivityEncoded'
# Transform the labels from String to Integer via LabelEncoder
le = preprocessing.LabelEncoder()
# Add a new column to the existing DataFrame with the encoded values
df[LABEL] = le.fit_transform(df['activity'].values.ravel())
# Differentiate between test set and training set
df_test = df[df['user-id'] > 28]
df_train = df[df['user-id'] <= 28]
# Round numbers
df_train = df_train.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})
x_train, y_train = dp.create_segments_and_labels(df_train,
                                                 TIME_PERIODS,
                                                 STEP_DISTANCE,
                                                 LABEL)
print(x_train.shape)
print(y_train.shape)

# Set input & output dimensions
num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size
print(list(le.classes_))
input_shape = (num_time_periods * num_sensors)
x_train = x_train.reshape(x_train.shape[0], input_shape)
print('x_train shape:', x_train.shape)
print('input_shape:', input_shape)
'''
convert all feature data (x_train) and label data (y_train) 
into a datatype accepted by Keras.
'''
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
