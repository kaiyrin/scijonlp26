from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np



sbert_model = SentenceTransformer("all-MiniLM-L6-v2") #https://github.com/huggingface/sentence-transformers


def embeddings_1(csv_path: str, aim_path: str):

    df = pd.read_csv(csv_path)
    article_texts_1 = (df["title"] + " " + df["abstract"]).fillna("").tolist()

    # embeddings for articles ebeding as vectors here shape=(519, 384)
    articles_embeddings_1 = sbert_model.encode(
        article_texts_1, show_progress_bar=True, convert_to_numpy=True
    )
    np.save("data/article_embeddings.npy", articles_embeddings_1)
    print(articles_embeddings_1.shape)
    # load aim text
    with open(aim_path, "r", encoding="utf-8") as f:
        aim_text = f.read()

    aim_embedding = sbert_model.encode([aim_text], convert_to_numpy=True)[0]

    # emerge embeddings inside oriignal dataframe
    #df["embedding_1"] = pd.Series(list(articles_embeddings_1), index=df.index, dtype=object)
    #print(df.head(5))
    #print(df.shape)
    
    # save aim embedding separately (important!)
    np.save("data/aim_embedding.npy", aim_embedding)

   
   
    return articles_embeddings_1, aim_embedding





if __name__ == "__main__":
    embeddings_1("data/sampled_lancet_psychiatry_1000.csv", "data/aim_scope.txt")