import datetime as dt
from builtins import KeyError, AttributeError
import hrvanalysis as hrv
import pandas as pd


# dataRead() verifies the input data for error free
def data_read(file):
    try:
        df = pd.read_csv(file, sep=',')
        # Round data frame values to 2 decimls after '.'.
        df = df.round(2)
        # Clean the data from Nan ,missing and duplicate values
        df = df.dropna(axis=0, how='any')
        cols = df.columns
        for col in cols:
            if col == 'TIME':
                df = df[df.TIME != '0']
                # Select duplicate rows and drop them based on time
                df = df.drop_duplicates(subset=['TIME'])
                # Extracting time in mm:ss format
                if type(df['TIME']) == str:
                    time = df['TIME'].tolist()
                    min_sec = []
                    for i in time:
                        m, s = i.split(':')
                        s = int(float(s))
                        min_sec.append(dt.datetime.strptime(m + ':' + str(s), '%M:%S').time())
                        df['TIME'] = min_sec
                # df = df.set_index('TIME', inplace=False)
            elif col == 'HR':
                df = df[df.HR > 0]
            elif col == 'GSR':
                df = df[df.GSR > 0.0]
    except pd.errors.EmptyDataError:
        print("No columns to parse from file,please re-upload again!")
    except KeyError:
        raise KeyError('Missing data! Please check HR,GSR and TimeStamp exist')
    except AttributeError:
        print('Missing data! Please check HR,GSR and TimeStamp data exist')
    return df


# Extracting the RR intervals HR(BPM) ==> HRV(ms): 1bpm =60,000 ms
def rr_process(hr):
    rr = []
    for h in hr:
        rr.append(int((60000 / h)))
    # This remove outliers from signal
    rr_without_outliers = hrv.remove_outliers(rr_intervals=rr, low_rri=300, high_rri=2000)
    # This replace outliers nan values with linear interpolation
    interpolated_rr_intervals = hrv.interpolate_nan_values(rr_intervals=rr_without_outliers,
                                                           interpolation_method="linear")
    rr = interpolated_rr_intervals
    # This remove ectopic beats from signal
    nn_intervals_list = hrv.remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="malik")
    # This replace ectopic beats nan values with linear interpolation
    interpolated_nn_intervals = hrv.interpolate_nan_values(rr_intervals=nn_intervals_list)
    rr = interpolated_nn_intervals
    return rr
