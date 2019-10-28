from collections import Counter

import pandas as pd
import tabulate as tb
df=pd.read_csv('Data/RR.csv',sep=',')


nn = Counter(df['RR'])
mode=nn.most_common(1)[0][1]
size = len(df['RR'])
amo= ((mode/size)*100)
mxmn = df['RR'].max() - df['RR'].min()
si = (amo/(2*mode*mxmn))
print(si)
print(tb.tabulate(df,headers="firstrow", tablefmt='github'))
