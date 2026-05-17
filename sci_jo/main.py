import os, sys
try:
    from sci_jo.data_loader import load_pubmed_articles
except ImportError as e:
    print(f"Error importing data_loader: {e}")
    sys.exit(1)
from sci_jo.data_loader import aim_load
from sci_jo.embedding import create_embeddings
def main():
    df = load_pubmed_articles()

    if df is None:
        return

    print(df.head())
    print(df.shape)

    df.to_csv("data/sampled_lancet_psychiatry_1000.csv", index=False)
    aim_str = aim_load("data/aim_scope.txt")
    print(aim_str[:500])  # Print the first 500 characters of the loaded
    create_embeddings("data/sampled_lancet_psychiatry_1000.csv", "data/aim_scope.txt")
    
if __name__ == "__main__":
    main()