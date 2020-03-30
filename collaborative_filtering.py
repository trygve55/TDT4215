import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def get_neighbourhood(ratings, k, item_based):
    if item_based:
        ratings = ratings.T

    cosine_sim = cosine_similarity(ratings, ratings)  # Cosine sim instead of linear kernel to get normalized values
    ind = np.flip(np.argsort(cosine_sim, axis=1), 1)  # Get the indices of the most similar users or items for each users sorted
    ind_k = np.delete(ind, 0, 1)  # The first row corresponds to itself so remove it
    ind_k = np.delete(ind_k, slice(k, None), 1)  # We are only looking at the k most similar, so remove to less similar ones
    return cosine_sim, ind_k


def collaborative_filtering(ratings, k, item_based=False):
    cosine_sim, ind_k = get_neighbourhood(ratings, k, item_based)
    return item_based_collab_filtering(ratings, k, cosine_sim, ind_k) if item_based\
        else user_based_collab_filtering(ratings, k, cosine_sim, ind_k)


def user_based_collab_filtering(ratings, k, cosine_sim, ind_k):
    recommendations = np.empty((ratings.shape[0], k))
    for user in tqdm(range(ratings.shape[0])):  # Find recommendations for every user
        relevant_documents = []
        for document in range(ratings.shape[1]):  # Look at every document to see how relevant it is
            if ratings[user, document] == 0:  # If the user has seen the document before, it should not be recommended
                value = 0
                for sim_user in ind_k[user]:  # Look at the k most similar users
                    if ratings[sim_user, document] == 1:  # If the similar user has seen the document it is likely relevant to the current user
                        value += cosine_sim[user, sim_user]  # Add the similarity value between the users as the value used to rank the document
                relevant_documents.append((document, value))  # Append both the document index and the value for the document
        relevant_documents = sorted(relevant_documents, key=lambda x: x[1], reverse=True)  # Sort the documents based on the value
        relevant_documents = relevant_documents[:k]  # Keep only the k most relevant
        relevant_documents = [i for i, j in relevant_documents]  # Drop the value to keep only the documents
        recommendations[user] = np.array(relevant_documents)
    return recommendations


def item_based_collab_filtering(ratings, k, cosine_sim, ind_k):
    recommendations = np.empty((ratings.shape[0], k))
    for user in tqdm(range(ratings.shape[0])):  # Find recommendations for every user
        relevant_documents = []
        for document in range(ratings.shape[1]):  # Look at every document to see how relevant it is
            if ratings[user, document] == 0:  # If the user has seen the document before, it should not be recommended
                value = 0
                for sim_article in ind_k[document]:  # Look at the k most similar articles
                    if ratings[user, sim_article] == 1:  # If the user has seen the similar article it is likely relevant to the current user
                        value += cosine_sim[document, sim_article]  # Add the similarity value between the documents as the value used to rank the document
                relevant_documents.append((document, value))  # Append both the document index and the value for the document
        relevant_documents = sorted(relevant_documents, key=lambda x: x[1], reverse=True)  # Sort the documents based on the value
        relevant_documents = relevant_documents[:k]  # Keep only the k most relevant
        relevant_documents = [i for i, j in relevant_documents]  # Drop the value to keep only the documents
        recommendations[user] = np.array(relevant_documents)
    return recommendations
