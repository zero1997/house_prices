import pandas as pd
import numpy as np 

 
 
# df = pd.DataFrame({'key1':['a', 'a', 'b', 'b', 'a'], 'key2':['one', 'two', 'one', 'two', 'one'],
#                 'data1':np.random.randn(5),
#                 'data2':np.random.randn(5)})

'''
Name	Brand	Cloth	Count
girl	uniql	sweater	3
girl	etam	suit	1
girl	etam	pants	1
girl	lagogo	jacket	2
boy		        pants	2
boy	    hailan	t-shirt	1
mother	hengyuanxiang	coat	2
mother	hengyuanxiang	sweater	1
mother		    coat	1
father	hailan	t-shirt	2
father	hailan	sweater	1
father	hailan	pants	3
'''

df = pd.DataFrame({'Name': ['girl', 'girl', 'girl', 'girl', 'boy', 'boy', 'mother', 'mother', 'mother', 'father', 'father', 'father'],
                  'Brand':['uniql', 'etam', 'etam', 'lagogo', None, 'hailan', 'hengyuanxiang', 'hengyuanxiang', None, 'hailan', 'hailan', 'hailan'],
                  'Cloth':['sweater', 'suit', 'pants', 'jacket', 'pants', 't-shirt', 'coat',  'sweater', 'coat', 't-shirt', 'sweater', 'pants'],
                  'Count':[3, 1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 3]})
# print df.groupby('Brand').sum()
print df.groupby(['Name', 'Brand'])['Count'].count()