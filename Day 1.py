import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset('tips')
print(sns.lineplot(y = 'total_bill', x = 'tip', data = tips))
print(sns.lineplot(y = 'total_bill', x = 'size', data = tips))

plt.show()