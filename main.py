import pandas as pd
from pathlib import Path

from data_loader import load_pubmed_articles
from embedding import embeddings_1
from similarity import compute_similarity
from bertopic_articles import (
    umap_articles,
    hdbscan_articles,
    bertopic_articles,
)

DATA_DIR = Path("data")
ARTICLES_CSV = DATA_DIR / "sampled_lancet_psychiatry_all.csv"
SAMPLED_ARTICLES_CSV = DATA_DIR / "sampled_lancet_psychiatry_1000.csv"
WORKING_DATA = DATA_DIR / "sampled_lancet_psychiatry_1000_preprcessing_skitlearn.csv"
AIM_TXT = DATA_DIR / "aim_scope.txt"
ARTICLE_EMBEDDINGS = DATA_DIR / "article_embeddings.npy"
AIM_EMBEDDING = DATA_DIR / "aim_embedding.npy"
SIMILARITY_CSV = DATA_DIR / "article_similarity_scores.csv"
"""
def run_pipeline():

    #articles_df = pd.read_csv(SAMPLED_ARTICLES_CSV)
    articles_df = pd.read_csv(WORKING_DATA)
    

    docs = (
        articles_df["title"].fillna("")
        + " "
        + articles_df["abstract"].fillna("")
    ).tolist()

    article_embeddings, aim_embedding = embeddings_1(
        csv_path=str(SAMPLED_ARTICLES_CSV),
        aim_path=str(AIM_TXT),
        article_embeddings_path=str(ARTICLE_EMBEDDINGS),
        aim_embedding_path=str(AIM_EMBEDDING),
    )
    similarity_scores = compute_similarity(
        str(ARTICLE_EMBEDDINGS), str(AIM_EMBEDDING)
    )
    umap_model, cluster_embeddings, aim_cluster, viz_embeddings, aim_viz, _ = (
        umap_articles(
            article_embeddings,
            aim_embedding
        )
    )

    hdbscan_model, cluster_labels = (
        hdbscan_articles(cluster_embeddings)
    )
    bertopic_model, topics, probs = bertopic_articles(
        umap_model_results=umap_model,
        hdbscan_model_results=hdbscan_model,
        docs=docs,
        article_embeddings=article_embeddings
    )
    return {
        "articles": articles_df,
        "docs": docs,
        "article_embeddings": article_embeddings,
        "aim_embedding": aim_embedding,
        "similarity_scores": similarity_scores,
        "cluster_embeddings": cluster_embeddings,
        "cluster_labels": cluster_labels,
        "viz_embeddings": viz_embeddings,
        "aim_viz": aim_viz,
        "umap_model": umap_model,
        "hdbscan_model": hdbscan_model,
        "bertopic_model": bertopic_model,
        "topics": topics,
        "probs": probs,
    }
"""
def main():
    #run_pipeline()
    #DATA LOADING
    articles_df = pd.read_csv(
        WORKING_DATA
        ) #checked that this is the same as the one used for embeddings and similarity
    docs = (articles_df["title"] + " " + articles_df["abstract"]).fillna("").tolist()
    print(f"Loaded articles: {articles_df.shape} |  Head: {articles_df.head(2)}")

    #EMBEDDING
    article_embeddings, aim_embedding = embeddings_1(
        csv_path=str(WORKING_DATA),
        aim_path=str(AIM_TXT),
        article_embeddings_path=str(ARTICLE_EMBEDDINGS),
        aim_embedding_path=str(AIM_EMBEDDING),
    )
    print(f"Article embeddings: {article_embeddings.shape}")

    #CHECK MISSMATCH BETWEEN DOCS AND EMBEDDINGS
    print(f"Docs length: {len(docs)} |  Embeddings shape: {article_embeddings.shape}")
    print(docs[:3])
    print(article_embeddings[:3])
    assert len(docs) == article_embeddings.shape[0], "docs and embeddings must match 1:1" #to make sure


    #SIMILARITY
    sim1 = compute_similarity(str(ARTICLE_EMBEDDINGS), str(AIM_EMBEDDING))
    #pmid = pd.read_csv(str(SAMPLED_ARTICLES_CSV))["pmid"].tolist()
    summary_df = pd.DataFrame({"pmid": articles_df["pmid"], "similarity_to_aim": sim1})
    print("\nArticles similarity to aim:", summary_df.head())

    #UMAP
    umap_model, cluster_embeddings, _, _, _, _ = umap_articles(
    article_embeddings,
    aim_embedding
    )   
    hdbscan_model_explore, cluster_labels = hdbscan_articles(cluster_embeddings)
    n_topics = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = (cluster_labels == -1).sum()
    print(f"[HDBSCAN explore] Topics: {n_topics}  |  Noise: {n_noise}")



    hdbscan_model, cluster_labels = (
        hdbscan_articles(cluster_embeddings)
        )
    print(f"[HDBSCAN] Cluster labels : {cluster_labels.shape}")
    n_topics = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise  = (cluster_labels == -1).sum()
    print(f"[HDBSCAN] Topics found : {n_topics}  |  Noise points : {n_noise}")  

    #BERTopic
    topic_model, topics, probs = bertopic_articles(
    umap_model,
    hdbscan_model,
    docs,
    article_embeddings
    )
    if topics is not None:
        print(f"BERTopic topics: {len(set(topics)) - (1 if -1 in topics else 0)}")


    return {
        "articles": articles_df,
        "docs": docs,
        "article_embeddings": article_embeddings,
        "aim_embedding": aim_embedding,
        "similarity_scores": sim1,
        "cluster_embeddings": cluster_embeddings,
        "cluster_labels": cluster_labels,
        "umap_model": umap_model,
        "hdbscan_model": hdbscan_model,
        "bertopic_model": topic_model,
        "topics": topics,
        "probs": probs
    }
    

if __name__ == "__main__":
    main()