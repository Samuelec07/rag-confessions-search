#RAG Document Search System (Saint Augustine's Confessions)

A Retrieval-Augmented Generation (RAG) search engine built with Python, LangChain, and ChromaDB. This project processes large text documents, splits them into semantic chunks, generates vector embeddings, and performs fast similarity searches for user queries.

#Features
- Semantic Vector Search: Powered by `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace).
- Vector Database: Local persistence using **ChromaDB**.
- Document Chunking: Uses `RecursiveCharacterTextSplitter` with 1000-character chunks and 500-character overlap.
- Privacy & Cost-Free: Runs 100% locally without external API dependencies or costs.

#Tech Stack
- **Language:** Python 3.10+
- **Frameworks:** LangChain, LangChain-Chroma, LangChain-HuggingFace
- **Vector Store:** ChromaDB
- **Data Source:** *The Confessions of Saint Augustine* (Public Domain)

#Installation & Setup

1. Clone the repository:
   ```bash
   git clone [https://github.com/Samuelec07/rag-confessions-search.git](https://github.com/Samuelec07/rag-confessions-search.git)
   cd rag-confessions-search
