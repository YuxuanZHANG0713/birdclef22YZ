import pandas as pd
import numpy as np
def get_code():
    train = pd.read_csv('train.csv')
    uniq = train.primary_label.unique()
    d1 = dict((i,j) for i,j in zip(uniq, range(np.size(uniq))))

    return d1

print(get_code())
