# A short primer on Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) is a technique that gives a language model access to external knowledge at answer time. Instead of relying only on what the model memorised during training, a RAG system first retrieves relevant passages from a document collection and then asks the model to answer using those passages.

## Why RAG is useful

Language models can hallucinate, meaning they produce fluent but incorrect statements. Grounding the answer in retrieved passages reduces hallucination and lets the system cite its sources. RAG also avoids expensive retraining: updating the knowledge base only requires re-indexing documents.

## The pipeline

1. Ingestion: documents are loaded, cleaned and split into chunks of a few hundred words, usually with some overlap so ideas that cross a boundary are not lost.
2. Embedding: each chunk is converted into a dense vector by an embedding model such as all-MiniLM-L6-v2, which produces 384-dimensional vectors.
3. Indexing: the vectors are stored in a vector index such as FAISS, which supports fast nearest-neighbour search.
4. Retrieval: the user's question is embedded and the most similar chunks are returned using cosine similarity.
5. Generation: the retrieved chunks are placed in the prompt and the language model writes an answer that cites them.

## Hybrid search

Dense vectors capture meaning but can miss exact keywords such as product codes, names or numbers. BM25 is a classic keyword ranking function that excels at exact matches. Hybrid search runs both and merges the rankings, commonly with Reciprocal Rank Fusion (RRF), which scores each document as the sum of 1 divided by (k plus its rank) across rankers, with k typically set to 60.

## Reranking

A cross-encoder reranker reads the question and a candidate passage together and outputs a relevance score. It is more accurate than the embedding model but much slower, so it is applied only to the top 20 or so candidates returned by first-stage retrieval.

## Evaluating RAG

Good evaluation separates retrieval from generation. Retrieval is measured with hit rate at k and mean reciprocal rank (MRR). Generation is measured with faithfulness, which checks that every claim in the answer is supported by the retrieved context, answer relevancy, and answer correctness against a reference. Frameworks such as RAGAS popularised these metrics.
