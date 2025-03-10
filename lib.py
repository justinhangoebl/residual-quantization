import numpy as np
import pandas as pd

def create_interaction_matrix(dataset_name, separator='\t'):
    df = pd.read_csv(dataset_name, sep=separator)
    
    interactions = np.zeros((max(df['user_id'])+1, max(df['item_id'])+1))

    interactions[df['user_id'], df['item_id']] = df['count']
    return interactions