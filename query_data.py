import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "chroma"

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)


    results = db.similarity_search_with_score(query_text, k=3)

    print(f"\n Suche nach: '{query_text}'")
    print("=" * 60)

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n Ergebnis {i} (Score / Distanz: {score:.4f}):")
        print(f"Quelle: {doc.metadata.get('id', 'Unbekannt')}")
        print("-" * 40)
        print(doc.page_content.strip())
        print("=" * 60)

if __name__ == "__main__":
    main()