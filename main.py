import os
import numpy as np
import pandas as pd
from pathlib import Path

from data_loader import load_pubmed_articles
from embedding import embeddings_1
from similarity import compute_similarity
from bertopic_articles import  umap_articles,hdbscan_articles, bertopic_articles


DATA_DIR = Path("data")
ARTICLES_CSV = DATA_DIR / "sampled_lancet_psychiatry_all.csv"
SAMPLED_ARTICLES_CSV = DATA_DIR / "sampled_lancet_psychiatry_1000.csv"
AIM_TXT = DATA_DIR / "aim_scope.txt"
ARTICLE_EMBEDDINGS = DATA_DIR / "article_embeddings.npy"
AIM_EMBEDDING = DATA_DIR / "aim_embedding.npy"
SIMILARITY_CSV = DATA_DIR / "article_similarity_scores.csv"


def main():

    #DATA LOADING
    articles_df = load_pubmed_articles(output_csv_path=str(ARTICLES_CSV))
    print(f"Loaded articles: {articles_df.shape} |  Head: {articles_df.head(2)}")

    #EMBEDDING
    article_embeddings, aim_embedding = embeddings_1(
        csv_path=str(SAMPLED_ARTICLES_CSV),
        aim_path=str(AIM_TXT),
        article_embeddings_path=str(ARTICLE_EMBEDDINGS),
        aim_embedding_path=str(AIM_EMBEDDING),
    )
    print(f"Article embeddings: {article_embeddings.shape}")



    #SIMILARITY
    sim1 = compute_similarity(str(ARTICLE_EMBEDDINGS), str(AIM_EMBEDDING))
    pmid = pd.read_csv(str(SAMPLED_ARTICLES_CSV))["pmid"].tolist()
    summary_df = pd.DataFrame({"pmid": pmid, "similarity_to_aim": sim1})
    print("\n Articles similarity to aim:", summary_df.head())

    #UMAP
    umap_result =umap_articles(str(ARTICLE_EMBEDDINGS), str(AIM_EMBEDDING))
    hdbscan_result = hdbscan_articles(umap_result.cluster_embeddings)
    n_topics = len(set(hdbscan_result.cluster_labels)) - (
        1 if -1 in hdbscan_result.cluster_labels else 0
    )
    n_noise = int((hdbscan_result.cluster_labels == -1).sum())
    print(f"\nHDBSCAN topics: {n_topics} | noise points: {n_noise}")    
    docs = build_article_texts(pd.read_csv(str(SAMPLED_ARTICLES_CSV)))
    bertopic_result = bertopic_articles(umap_result.cluster_embeddings, hdbscan_result.cluster_labels, umap_result.viz_embeddings, umap_result.aim_viz)
    print(f"BERTopic topics: {len(set(bertopic_result.topics))}")   
    


if __name__ == "__main__":
    main()