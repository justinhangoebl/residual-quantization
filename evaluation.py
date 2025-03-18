import numpy as np


def dcg_at_k(relevance_scores, k):
    relevance_scores = np.asarray(relevance_scores)[:k]
    if relevance_scores.size == 0:
        return 0.0
    
    discounts = np.log2(np.arange(2, relevance_scores.size + 2))
    return np.sum(relevance_scores / discounts)


def idcg_at_k(relevance_scores, k):
    
    ideal_relevance = np.sort(relevance_scores)[::-1]
    return dcg_at_k(ideal_relevance, k)


def ndcg_at_k(relevance_scores, k):
    
    if len(relevance_scores) == 0:
        return 0.0
    
    dcg = dcg_at_k(relevance_scores, k)
    idcg = idcg_at_k(relevance_scores, k)
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_ndcg(predictions, true_ratings, k=10):
    relevance_scores = [true_ratings[item_idx] for item_idx in predictions[:k]]
    return ndcg_at_k(relevance_scores, k)


def evaluate_recommendations(recommender, test_data, k=10, users_to_evaluate=None):
    if users_to_evaluate is None:
        users_to_evaluate = range(test_data.shape[0])
    
    ndcg_scores = []
    
    for user_idx in users_to_evaluate:
        true_ratings = test_data[user_idx].cpu().numpy()
        
        recommendations = recommender.recommend_items(test_data, user_idx, top_k=k, exclude_known=False)
        recommended_items = [item_idx for item_idx, _ in recommendations]
        
        ndcg = calculate_ndcg(recommended_items, true_ratings, k)
        ndcg_scores.append(ndcg)
        
    return np.mean(ndcg_scores)