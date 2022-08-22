from telnetlib import PRAGMA_HEARTBEAT
import pandas as pd
import numpy as np
df = pd.DataFrame(dict(x=['A', 'B', 'C', 'A','B','C'], year=[2010,2010,2010,2011,2011,2011],
                        value=[1,3,4,3,5,2]))

df = df.set_index(['x', 'year']) #设置索引

print(df.loc[('A', 2010), 'value']) # loc取值
print(df.loc[('A', slice(None)), 'value']) # 切片，代表任何值

a = ['a','b']
b = [1,2]
x,y = np.meshgrid(a,b)
print(pd.DataFrame({'x':x.flatten(), 'y':y.flatten()}))

df = pd.DataFrame({'x':['a','b','c'], '2010':[1,3,4], '2011':[3,5,2]})
df_melt = pd.melt(df, id_vars='x', var_name='year', value_name='value') # 多个属性合并成1个属性
df_melt.reset_index()
df_melt = df_melt.pivot_table()
print(df_melt)
print(df)