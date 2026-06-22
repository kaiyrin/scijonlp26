from embedding import embeddings_1
import umap.umap_ as umap
import hdbscan
import numpy as np
from bertopic import BERTopic


def umap_articles(article_emb_path: str, aim_emb_path: str):  #code source https://umap-learn.readthedocs.io/en/latest/basic_usage.html#penguin-data
    article_embeddings = np.load(article_emb_path) if isinstance(article_emb_path, (str, bytes, np.str_, np.bytes_)) else article_emb_path
    aim_embedding = np.load(aim_emb_path) if isinstance(aim_emb_path, (str, bytes, np.str_, np.bytes_)) else aim_emb_path

    aim_matrix = aim_embedding.reshape(1, -1) #(0,1)  # reshape for sklearn because journal size (519,384) aim size (384,) needs to be (1,384 for cosine_similarity to work)
    all_vectors = np.vstack([article_embeddings, aim_matrix])#(520,384)
    reducer_2d = umap.UMAP(n_components=2, random_state=42, metric="cosine")  #https://umap-learn.readthedocs.io/en/latest/auto_examples/plot_algorithm_comparison.html
    all_2d = np.asarray(reducer_2d.fit_transform(all_vectors))  # (522, 2)
    all_reduced_visualization = reducer_2d.fit_transform(all_vectors)  # (522, 2)
    reducer_5d = umap.UMAP(n_components=5, random_state=42)  #https://umap-learn.readthedocs.io/en/latest/auto_examples/plot_algorithm_comparison.html
    all_5d = np.asarray(reducer_5d.fit_transform(all_vectors))  # (522, 5)      
    
    # Split articles vs aim
    cluster_embeddings = all_5d[:-1]   # (N, 5)
    aim_cluster        = all_5d[-1]    # (5,)
    viz_embeddings     = all_2d[:-1]       # (N, 2)
    aim_viz            = all_2d[-1]
    print(f"[UMAP] Cluster projection : {cluster_embeddings.shape}")
    print(f"[UMAP] Viz projection     : {viz_embeddings.shape}")

    return reducer_5d, cluster_embeddings, aim_cluster, viz_embeddings, aim_viz


def hdbscan_articles(cluster_embeddings, aim_cluster, viz_embeddings, aim_viz): #https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean", cluster_selection_method="eom", gen_min_span_tree=True)
    #parametres
    
    # HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    # gen_min_span_tree=True, leaf_size=40, memory=Memory(None),
    # metric='euclidean', min_cluster_size=5, min_samples=None, p=None)
 
    cluster_labels = clusterer.fit_predict(cluster_embeddings)
    print(f"[HDBSCAN] Cluster labels : {cluster_labels.shape}")
    n_topics = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise  = (cluster_labels == -1).sum()
    print(f"[HDBSCAN] Topics found : {n_topics}  |  Noise points : {n_noise}")
    return cluster_labels

def bertopic_articles(cluster_embeddings, cluster_labels, viz_embeddings, aim_viz):
    umap_model = umap_articles.reducer_5d
    hdbscan_model = hdbscan_articles.clusterer
   
    topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    language="english",
    calculate_probabilities=True)

    topics, probs = topic_model.fit_transform(docs, embeddings=article_embeddings)
    print(f"[BERTopic] Topics : {len(set(topics))}  |  Probabilities : {probs.shape}")
    return topics, probs

if __name__ == "__main__":
    umap_articles(article_emb_path, aim_emb_path)
