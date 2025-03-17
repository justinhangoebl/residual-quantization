import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def create_interaction_matrix(df, column_name='user_id', row_name='item_id', value_name='count'):
    
    interactions = np.zeros((max(df[column_name])+1, max(df[row_name])+1))

    if value_name == None:
        interactions[df[column_name], df[row_name]] = 1
    else:
        interactions[df[column_name], df[row_name]] = df[value_name]
    return interactions

def generate_splits(training_data: str|list[str], sep='\t', random_state=42, train_size=0.6, val_size=0.2):
    if type(training_data) == 'str':
        data = pd.read_csv(training_data, sep=sep)
    else:
        data = [pd.read_csv(file, sep=sep) for file in training_data]
        data = pd.concat(data)
        
    train, validate, test = np.split(data.sample(frac=1, random_state=random_state), [int(train_size*len(data)), int((train_size+val_size)*len(data))])
    
    return train, validate, test