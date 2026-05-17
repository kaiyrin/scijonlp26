import os
import numpy as np
import pandas as pd

from sci_jo.data_loader import load_pubmed_articles
from sci_jo.embedding import embeddings_1
from sci_jo.similarity import compute_similarity


def main():

    # =========================================================
    # 1. LOAD DATA
    # =========================================================
    df = load_pubmed_articles()

    print("\n[STEP 1] Data loaded")
    print(df.shape)
    print(df.head(2))


    # =========================================================
    # 2. EMBEDDINGS
    # =========================================================
    article_embeddings, aim_embedding = embeddings_1("data/sampled_lancet_psychiatry_1000.csv", "data/aim_scope.txt")

    print("\n[STEP 2] Embeddings created")
    print(article_embeddings.shape)


    # attach embeddings to df (optional but useful)
    df["embedding"] = list(article_embeddings)


    # =========================================================
    # 3. SIMILARITY
    # =========================================================
    sims = compute_similarity(article_embeddings, aim_embedding)

    # ensure correct shape
    sims = np.array(sims).flatten()

    df["similarity_to_aim"] = sims

    print("\n[STEP 3] Similarity computed")
    print(df[["pmid", "similarity_to_aim"]].head())


    # =========================================================
    # 4. SAVE FINAL DATASET
    # =========================================================
    os.makedirs("data", exist_ok=True)

    output_path = "data/final_lancet_psychiatry_dataset.csv"
    df.to_csv(output_path, index=False)

    print("\n[STEP 4] Saved final dataset:", output_path)
    print(df.shape)


if __name__ == "__main__":
    main()