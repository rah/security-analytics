import pandas as pd
import matplotlib.pyplot as plt
from pyattck import Attck

attack = Attck()
attack.update(enterprise=True)

techniques = []
data_sources = []

for technique in attack.enterprise.techniques:
    if technique.data_source:
        for data_source in technique.data_source:
            techniques.append(technique.name)
            data_sources.append(data_source)

data = {
    'technique': techniques,
    'data_source': data_sources
}

t2d = pd.DataFrame(data, columns=['technique', 'data_source'])

t2d.head(20)

# Look at the frequency of the data sources
dataFrequency = t2d['data_source'].value_counts()
dataFrequency.head(20)
plt.bar(dataFrequency.index, dataFrequency.values)
plt.xticks(dataFrequency.index, dataFrequency.index, rotation=90)

sorted_t2d = t2d.sort_values(['data_source'])

sorted_t2d.head(20)

sorted_t2d[sorted_t2d['data_source'] == "Process monitoring"]

len(t2d)
len(dataFrequency)

type(dataFrequency)

names = dataFrequency.index
