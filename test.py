import pandas as pd
 
 
df = pd.DataFrame([[1, 2, 3], [3, 4, 5], [5, 6, 7]],columns=['max_speed', 'shield', 'dsf'])
print df
df.loc[1, 'max_speed'] = None
df.loc[2, 'shield'] = None
print df.fillna(0)

