"""Generate shared evaluation answers once using the configured real RAG pipeline.

python -m backend.study.prepare_live --task study/live.example.json \
    --queries study/live-evaluation.example.json --out data/live-prepared.json
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from ..config import Settings
from ..ingestion import content_hash
from ..main import build_pipeline
from .live_materials import Document, Evaluation, load_materials


def prepare(task, queries, output, settings):
    materials = load_materials(task)
    pipeline = build_pipeline(settings)
    try:
        expected = {content_hash(d.text.encode())[:16] for d in materials.documents}
        existing = {d["doc_id"] for d in pipeline.store.documents()}
        if existing - expected:
            raise ValueError("Use a dedicated DATA_DIR containing only the study corpus")
        for document in materials.documents:
            pipeline.ingest(document.name, document.text.encode())
        evaluations = []
        for item in json.loads(Path(queries).read_text(encoding="utf-8")):
            result, answer, error = pipeline.answer(item["query"])
            if error or answer is None or result.confidence is None:
                raise RuntimeError(f"Could not prepare {item['id']}: {error}")
            evaluations.append(
                Evaluation(
                    id=item["id"],
                    query=item["query"],
                    answer=answer.text,
                    score=result.confidence.score,
                    answer_key=item.get("answer_key", ""),
                    passages=[
                        Document(
                            name=result.chunks[h.chunk_id].source,
                            text=result.chunks[h.chunk_id].text,
                        )
                        for h in result.fused
                    ],
                )
            )
        materials.evaluations = evaluations
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(materials.model_dump_json(indent=2), encoding="utf-8")
        print(f"Prepared {len(evaluations)} shared answers. Grade them before research use.")
    finally:
        pipeline.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    load_dotenv()
    prepare(args.task, args.queries, args.out, Settings.from_env())
