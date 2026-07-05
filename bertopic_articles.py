from embedding import embeddings_1
import umap.umap_ as umap
import hdbscan
import numpy as np
import pandas as pd
from bertopic import BERTopic


def umap_articles(article_embeddings, aim_embedding):  #code source https://umap-learn.readthedocs.io/en/latest/basic_usage.html#penguin-data 
    aim_matrix = aim_embedding.reshape(1, -1) #(0,1)  # reshape for sklearn because journal size (519,384) aim size (384,) needs to be (1,384 for cosine_similarity to work)
    all_vectors = np.vstack([article_embeddings, aim_matrix])#(520,384)
    reducer_2d = umap.UMAP(n_components=2, random_state=42, metric="cosine")  #https://umap-learn.readthedocs.io/en/latest/auto_examples/plot_algorithm_comparison.html
    all_2d = np.asarray(reducer_2d.fit_transform(all_vectors))  # (522, 2)
    all_reduced_visualization = reducer_2d.fit_transform(all_vectors)  # (522, 2)
    reducer_5d = umap.UMAP(n_components=5, random_state=42, metric="cosine")  #https://umap-learn.readthedocs.io/en/latest/auto_examples/plot_algorithm_comparison.html
    all_5d = np.asarray(reducer_5d.fit_transform(all_vectors))  # (522, 5)      
    
    # Split articles vs aim
    cluster_embeddings = all_5d[:-1]   # (N, 5)
    aim_cluster        = all_5d[-1]    # (5,)
    viz_embeddings     = all_2d[:-1]       # (N, 2)
    aim_viz            = all_2d[-1]
    print(f"[UMAP] Cluster projection : {cluster_embeddings.shape}")
    print(f"[UMAP] Viz projection     : {viz_embeddings.shape}")
    all_2d_df = pd.DataFrame(all_2d, columns=["UMAP1", "UMAP2"])
    all_2d_df.to_csv("data/viz_embeddings_2d.csv", index=False)
    print(f"[UMAP] Saved viz embeddings to data/viz_embeddings_2d.csv")

    return reducer_5d, cluster_embeddings, aim_cluster, all_reduced_visualization


def hdbscan_articles(cluster_embeddings): #https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, 
                                metric="euclidean", 
                                cluster_selection_method="eom", 
                                gen_min_span_tree=True,
                                prediction_data=True)  #otherwise i got error wwith "No predition data was generated" WARNING used AI for debugging
    #parametres
    
    # HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
    # gen_min_span_tree=True, leaf_size=40, memory=Memory(None),
    # metric='euclidean', min_cluster_size=5, min_samples=None, p=None)
 
    cluster_labels = clusterer.fit_predict(cluster_embeddings)
    return clusterer, cluster_labels


def bertopic_articles(umap_model_results, hdbscan_model_results, docs, article_embeddings):
    
    topic_model = BERTopic(
        umap_model=umap_model_results,
        hdbscan_model=hdbscan_model_results,
        language="english",
        calculate_probabilities=True) 

    topics, probs = topic_model.fit_transform(docs, embeddings=article_embeddings)
    print(f"[BERTopic] Topics : {len(set(topics))}  |  Probabilities : {probs.shape}")
    return topic_model, topics, probs

if __name__ == "__main__":
    # Load article embeddings and aim embedding
    article_embeddings = np.load("data/article_embeddings.npy")
    aim_embedding = np.load("data/aim_embedding.npy")

    # Perform UMAP dimensionality reduction
    umap_result = umap_articles(article_embeddings, aim_embedding)
    _, cluster_embeddings, aim_cluster, all_reduced_visualization = umap_result

    # Save viz embeddings and aim_viz to CSV
   

    # Perform HDBSCAN clustering
    hdbscan_model, cluster_labels = hdbscan_articles(cluster_embeddings)
    print(f"[HDBSCAN] Cluster labels : {cluster_labels.shape}")
    n_topics = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise  = (cluster_labels == -1).sum()
    print(f"[HDBSCAN] Topics found : {n_topics}  |  Noise points : {n_noise}")  

    # Perform BERTopic modeling
    bertopic_result = bertopic_articles(umap_model_results=umap_result[0], 
                                        hdbscan_model_results=hdbscan_model, 
                                        docs=pd.read_csv("data/sampled_lancet_psychiatry_1000.csv")["title"].tolist(), 
                                        article_embeddings=article_embeddings) 
    print(f"[BERTopic] Topics found : {len(set(bertopic_result[1]))}")

