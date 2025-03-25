import pandas as pd
import numpy as np

def inter_matr_implicit(users: pd.DataFrame,
                        items: pd.DataFrame,
                        interactions: pd.DataFrame,
                        threshold=1) -> np.ndarray:
    res = None
    interactions = interactions.copy()
    n_users = len(users.index)
    n_items = len(items.index)

    res = np.zeros([n_users, n_items], dtype=np.int8)

    inter_column_name = 'count'

    row = interactions['user_id'].to_numpy()
    col = interactions["item_id"].to_numpy()

    data = interactions[inter_column_name].to_numpy()
    data[data < threshold] = 0
    data[data >= threshold] = 1

    res[row, col] = data

    return res
