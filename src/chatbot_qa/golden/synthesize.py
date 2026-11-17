"""Generate synthetic edge-case goldens from the knowledge base with DeepEval's Synthesizer.

    python -m chatbot_qa.golden.synthesize --per-context 2

The output goes to ``datasets/synthetic/`` and is NOT loaded by the test suite.
A person reviews each item (is the expected answer right? is the question
realistic?) and moves the approved ones into ``datasets/golden/rag_<name>.json``.
Unreviewed synthetic data must never become a release criterion.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from chatbot_qa.models import RagGolden


def load_contexts(knowledge_dir: Path) -> list[list[str]]:
    """One context (a list with one chunk) per knowledge-base file."""
    paths = sorted(knowledge_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No knowledge-base files (*.md) in {knowledge_dir}")
    return [[path.read_text(encoding="utf-8").strip()] for path in paths]


def build_synthesizer():
    from deepeval.synthesizer import Synthesizer

    model = os.getenv("EVAL_JUDGE_MODEL") or None
    try:
        # Bias generation towards harder questions: multi-step reasoning, added
        # constraints, hypotheticals and comparisons.
        from deepeval.synthesizer import Evolution
        from deepeval.synthesizer.config import EvolutionConfig

        evolution = EvolutionConfig(
            num_evolutions=1,
            evolutions={
                Evolution.REASONING: 0.3,
                Evolution.CONSTRAINED: 0.3,
                Evolution.HYPOTHETICAL: 0.2,
                Evolution.COMPARATIVE: 0.2,
            },
        )
        return Synthesizer(model=model, evolution_config=evolution)
    except (ImportError, TypeError):  # older DeepEval: fall back to its default evolutions
        return Synthesizer(model=model)


def to_rag_goldens(goldens: list, prefix: str) -> list[RagGolden]:
    converted = []
    for index, golden in enumerate(goldens, start=1):
        if not (golden.input and golden.expected_output and golden.context):
            continue
        converted.append(
            RagGolden(
                id=f"{prefix}-{index:03d}",
                input=golden.input,
                expected_output=golden.expected_output,
                expected_context=list(golden.context),
                tags=["synthetic", "needs-review"],
            )
        )
    return converted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--knowledge-dir", type=Path, default=Path("datasets/knowledge_base"))
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/synthetic"))
    parser.add_argument("--per-context", type=int, default=2, help="goldens to generate per knowledge file")
    args = parser.parse_args()

    contexts = load_contexts(args.knowledge_dir)
    synthesizer = build_synthesizer()
    generated = synthesizer.generate_goldens_from_contexts(
        contexts=contexts, include_expected_output=True, max_goldens_per_context=args.per_context
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    goldens = to_rag_goldens(generated, prefix=f"SYN-{stamp}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"rag_synthetic_{stamp}.json"
    out_path.write_text(
        json.dumps([g.model_dump() for g in goldens], indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(goldens)} synthetic goldens to {out_path}")
    print("Review them, then move the approved ones to datasets/golden/rag_<name>.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
