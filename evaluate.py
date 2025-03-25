import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
import rec
from pathlib import Path
import os.path

def get_ndcg_score_sk(df_predictions, test_interaction_matrix: np.ndarray, topK=10) -> float:
    score = None

    all_dcgs = np.zeros(len(df_predictions))
    for i, row in df_predictions.iterrows():
        g_truth = test_interaction_matrix[row["user"]]
        
        predicted_scores = np.zeros(len(g_truth))
        predictions = [int(e) for e in row["recs"].split(",")]

        for j, rec in enumerate(predictions):
            predicted_scores[rec] = topK - j + 2

        all_dcgs[i] = ndcg_score([g_truth], [predicted_scores], k=topK)

    score = np.average(all_dcgs)
    return score

def read(folder, dataset, file_ending, skip_rows=0):
    filename = f"{dataset}.{file_ending}"
    return pd.read_csv(os.path.join(folder, filename), sep='\t', skiprows=skip_rows)

def evaluate(file_path, topK=10):
    df_res = pd.read_csv(file_path, sep='\t', header=None, names=['user', 'recs'])

    dataset_path = 'evaluation-data'
    dataset_name = 'lfm-evaluation'
    users = read(dataset_path, dataset_name, 'user')
    items = read(dataset_path, dataset_name, 'item')
    test_inters = read(dataset_path, dataset_name, 'inter_secret_test')
    test_users = pd.read_csv('evaluation-data/test_users.txt').squeeze()

    test = rec.inter_matr_implicit(users=users, items=items, interactions=test_inters)

    test_data = test
    df_res_sorted = df_res.sort_values(by=["user"])
    df_sk_input = df_res_sorted[df_res_sorted.user.isin(test_users)].reset_index()
    new_result = get_ndcg_score_sk(df_sk_input, test_data, topK)

    return new_result


def evaluate_submission(res_file_path,
                        topK=10,
                        method="XXXXXX"):
    all_ndcg = evaluate(res_file_path, topK=topK)
    return {'method': method, 'ndcg': all_ndcg}


if __name__ == "__main__":
    res = []

    for dirpath, dirnames, filenames in os.walk("res"):
        for filename in [f for f in filenames if f.endswith(".tsv")]:
            print("Evaluating ", filename, "...")

            res_file = os.path.join(dirpath, filename)
            method = Path(res_file).parts[-1].split("_")[1]
            
            res.append(evaluate_submission(res_file, method=method))
    df = pd.DataFrame(res)
    print(df)
    df.to_csv("evaluation_results.tsv", sep='\t', index=False)

