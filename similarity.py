import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(article_emb_path:str, aim_emb_path:str):

    

    article_embeddings = np.load(article_emb_path)
    aim_embedding = np.load(aim_emb_path)

    # reshape for sklearn because journal size (519,384) aim size (384,) needs to be (1,384 for cosine_similarity to work)
    aim_embedding = aim_embedding.reshape(1, -1)

    # cosine similarity -> flatten to 1D for pandas
    sims1 = cosine_similarity(article_embeddings, aim_embedding).flatten()

    # ensure output directory exists
    import os
    os.makedirs("output_data", exist_ok=True)

    pd.DataFrame({
        "similarity_score": sims1
    }).to_csv("data/article_similarity_scores.csv", index=True)

    return sims1


if __name__ == "__main__":
    compute_similarity("data/article_embeddings.npy", "data/aim_embedding.npy"  )