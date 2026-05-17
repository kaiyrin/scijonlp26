import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(
    
    article_emb_path="data/article_embeddings.npy",
    aim_emb_path="data/aim_embedding.npy"
):

    

    article_embeddings = np.load(article_emb_path)
    aim_embedding = np.load(aim_emb_path)

    # reshape for sklearn
    aim_embedding = aim_embedding.reshape(1, -1)

    # cosine similarity
    sims1 = cosine_similarity(article_embeddings, aim_embedding)


    return  sims1.flatten()


if __name__ == "__main__":
    compute_similarity()