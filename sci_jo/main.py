from sci_jo.data_loader import load_pubmed_articles

def main():
    df = load_pubmed_articles()

    if df is None:
        return

    print(df.head())
    print(df.shape)

    df.to_csv("data/sampled_lancet_psychiatry_1000.csv", index=False)

if __name__ == "__main__":
    main()