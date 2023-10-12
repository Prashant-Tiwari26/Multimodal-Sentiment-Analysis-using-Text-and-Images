import os
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

    negative_images = os.listdir("Data/Images/Negative")
    neutral_images = os.listdir("Data/Images/Neutral")
    positive_images = os.listdir("Data/Images/Positive")

    def update_filename(row):
        filename = row['File Name']
        if filename in negative_images:
            return f"Negative/{filename}"
        elif filename in neutral_images:
            return f"Neutral/{filename}"
        elif filename in positive_images:
            return f"Positive/{filename}"
        else:
            return filename

    data['File Name'] = data['File Name'].str.replace('.txt', '.jpg')
    data['File Name'] = data.apply(update_filename, axis=1)

    data.drop(['Caption', 'Caption Length', 'Hashtags', 'Total Hashtags'], axis=1, inplace=True)

    data.to_csv("Data/Images/ImageLabelsSequenced.csv", index=False)

if __name__ == '__main__':
    EngineerFeatures()