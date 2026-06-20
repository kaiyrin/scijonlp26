import os
import numpy as np
import pandas as pd

from sci_jo.data_loader import load_pubmed_articles
from sci_jo.embedding import embeddings_1
from sci_jo.similarity import compute_similarity


def main():

    #DATA LOADING
    df = load_pubmed_articles()

    print("\n Data loaded")
    print(df.shape)
    print(df.head(2))


    #EMBEDDING
    article_embeddings, aim_embedding = embeddings_1("data/sampled_lancet_psychiatry_1000.csv", "data/aim_scope.txt")
    print(f"Article embeddings shape: {article_embeddings.shape}")



    #SIMILARITY
    article_emb_path = "data/article_embeddings.npy"
    aim_emb_path = "data/aim_embedding.npy"
    sim1 = compute_similarity(article_emb_path, aim_emb_path)
    pmid = pd.read_csv("data/sampled_lancet_psychiatry_1000.csv")["pmid"].tolist()
    summary_df = pd.DataFrame({"pmid": pmid, "similarity_to_aim": sim1})
    print("\n Articles similarity to aim:", summary_df.head())


    #UMAP
    


if __name__ == "__main__":
    main()