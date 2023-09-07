import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def EngineerFeatures():
    data = pd.read_excel("Data/LabeledText.xlsx")

    encode = LabelEncoder()
    data['LABEL'] = encode.fit_transform(data['LABEL'])
    '''
    0 -> Negative
    1 -> Neutral
    2 -> Positive
    '''
    data['Caption Length'] = data['Caption'].apply(len)
    data['Hashtags'] = data['Caption'].str.extractall(r'#(\w+)').groupby(level=0)[0].apply(' '.join)
    data['Hashtags'] = data['Hashtags'].fillna('')
    data['Total Hashtags'] = data['Hashtags'].str.split(' ').apply(lambda x: len(x) if x[0] != '' else 0)
    data['Hashtags'].replace('', np.nan, inplace=True)

    data.to_csv("Data/Text/Engineered.csv", index=False)

if __name__ == '__main__':
    EngineerFeatures()