import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df['Year_of_Release'] = df['Year_of_Release'].astype(int)
    return df
