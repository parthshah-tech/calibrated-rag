import pytest

from backend.config import Settings
from backend.embedder import HashEmbedder
from backend.pipeline import RAGPipeline
from backend.vector_store import InMemoryVectorStore

DOC_A = (
    "Reciprocal Rank Fusion combines rankings from several retrievers.\n\n"
    "Each document earns 1 over k plus its rank from every list it appears in.\n\n"
    "Fusion ignores raw scores because dense and sparse scores are not comparable."
)
DOC_B = (
    "BM25 is a sparse retrieval function based on term frequency and inverse document "
    "frequency.\n\nIt rewards exact keyword matches such as product code XJ-4471.\n\n"
    "Dense embeddings capture paraphrase and synonym similarity instead."
)
DOC_C = "Sourdough bread needs flour, water, salt and a live starter.\n\nBake it in a hot oven."


@pytest.fixture
def pipeline():
    s = Settings(chunk_strategy="recursive", chunk_size=120, chunk_overlap=20)
    return RAGPipeline(InMemoryVectorStore(), HashEmbedder(), s)


@pytest.fixture
def loaded(pipeline):
    for name, text in [("a.txt", DOC_A), ("b.txt", DOC_B), ("c.txt", DOC_C)]:
        pipeline.ingest(name, text.encode())
    return pipeline
