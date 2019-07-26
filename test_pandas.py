import pandas as pd
import numpy as np

data = {'Name':['Tom', 'nick', 'krish', 'jack', 'Tom', 'nick', 'krish', 'jack'], 'MiddleName':['Tom', 'nick', 'krish', 'jack', 'Tom', 'nick', 'krish', 'jack'], 'Age':[20, 21, 0.0000, 0, np.nan, np.nan, np.nan, np.nan]}


df = pd.DataFrame(data=data)

# boolean_array_no_nan_rows = df.iloc[:, -1].notnull()
# df = df[boolean_array_no_nan_rows]

for i, v in df.iterrows():
    print(f"{i}, {v}")

print("")