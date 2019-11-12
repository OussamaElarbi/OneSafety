import statistics as stats
import pandas as pd
import tabulate as tb
import hrvanalysis as hrv

df = pd.read_csv('/Users/smileyboy/OneSafety/Data/rr1.csv', sep=',')
# nn = Counter(df['RR'])
df = df.multiply(1000)
df = round(df, 0)
rr = df['RR'].tolist()
# This remove outliers from signal
rr = hrv.remove_outliers(rr_intervals=rr, low_rri=300, high_rri=2000)
mode = stats.mode(rr)
mode = rr.count(mode)
print(mode)
# mode=nn.most_common(1)[0][1]
size = len(rr)
print(size)
amo = ((mode / size) * 100)
print(amo)
print(max(rr))
print(min(rr))
mxmn = max(rr) - min(rr)
print(mxmn)
si = (amo / (2 * mode * mxmn))
df=pd.Series(rr).value_counts()
print(df/len(df))
print(si)
# print(tb.tabulate(df, headers="firstrow", tablefmt='github'))
