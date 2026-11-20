"""Bounded, persistent visitor collections with a shared embedding model.

The caller holds the app lock while obtaining and using a pipeline. Run one worker
so sparse indexes remain consistent with uploads and removals.
"""

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

from .embedder import build_embedder
from .llm import build_llm
from .pipeline import RAGPipeline
from .reranker import build_reranker
from .vector_store import ChromaVectorStore


class VisitorWorkspaces:
    def __init__(self, settings, capacity=8):
        self.settings = settings
        self.capacity = capacity
        self.pipelines = OrderedDict()
        self.embedder = None
        self.reranker = None

    def get(self, token):
        if token in self.pipelines:
            self.pipelines.move_to_end(token)
            return self.pipelines[token]
        if self.embedder is None:
            self.embedder = build_embedder(self.settings.embedder, self.settings.embed_model)
            self.reranker = build_reranker(self.settings.rerank_model)
        settings = replace(
            self.settings, data_dir=str(Path(self.settings.data_dir) / "visitors" / token)
        )
        pipeline = RAGPipeline(
            ChromaVectorStore(str(Path(self.settings.data_dir) / "visitor_chroma"), f"v_{token}"),
            self.embedder,
            settings,
            llm=build_llm(settings),
            reranker=self.reranker,
        )
        self.pipelines[token] = pipeline
        if len(self.pipelines) > self.capacity:
            _, evicted = self.pipelines.popitem(last=False)
            evicted.close()
        return pipeline

    def close(self):
        for pipeline in self.pipelines.values():
            pipeline.close()
