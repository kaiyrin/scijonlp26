import os, re
from dotenv import load_dotenv
import pandas as pd
from Bio import Entrez, Medline
from typing import Mapping, Any, Sequence
import random


load_dotenv()

NCBI_API_KEY = os.getenv("NCBI_API_KEY")
EMAIL = os.getenv("EMAIL")

Entrez.email = EMAIL
Entrez.api_key = NCBI_API_KEY

def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
    return text.strip()


def load_pubmed_articles(): 
    handle = Entrez.esearch(
        db="pubmed",
        term='''
        "Lancet Psychiatry"[Journal]
        AND ("2014/01/01"[PDAT] : "2026/05/01"[PDAT])
        AND english[Language]
        ''',
        retmax=5000
    )
    record = Entrez.read(handle)

    count = int(record.get("Count", 0))
    print(count)

    record_map: Mapping[str, Any] = record if isinstance(record, Mapping) else {}
    id_list = record_map.get("IdList")
    if not isinstance(id_list, Sequence) or isinstance(id_list, (str, bytes)) or len(id_list) == 0:
        raise ValueError("No PMC IDs returned from Entrez search")
    print(id_list[:5])  #first 5

    # sample 1000 IDs
    random.seed(42)
    sample_size = min(2000, len(id_list))
    sampled_ids = random.sample(id_list, sample_size)

    print("Sample size:", len(sampled_ids))
    print("First 5 sampled IDs:", sampled_ids[:5])

    # fetch sampled articles in batches
    rows = []

    for i in range(0, len(sampled_ids), 100):
        batch = sampled_ids[i:i + 100]
        fetch_handle = Entrez.efetch(
            db="pubmed",
            id=",".join(batch),
            rettype="medline",
            retmode="text"
        )
        records = Medline.parse(fetch_handle)

        for rec in records:

            title = rec.get("TI", "")
            abstract = rec.get("AB", "")

            if not title or not abstract:
                continue

            title = clean_text(title)
            abstract = clean_text(abstract)

            doi = ""

            for aid in rec.get("AID", []):

                if "[doi]" in aid.lower():

                    doi = aid.replace(" [doi]", "")
                    break

            rows.append({

                "pmid": rec.get("PMID", ""),
                "title": title,
                "abstract": abstract,
                "year": str(rec.get("DP", ""))[:4],
                "doi": doi,
                "authors": "; ".join(rec.get("AU", [])),
                "journal": rec.get("JT", "")

            })

    df = pd.DataFrame(rows)

    print(df.head())
    print("Final dataframe shape:", df.shape)

    df.to_csv("data/sampled_lancet_psychiatry_1000.csv", index=False)
    return df
def aim_load(file_path: str) -> str:
   with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
   return clean_text(text)
if __name__ == "__main__":
    load_pubmed_articles()
    aim_load("data/aim_scope.txt")