"""Draw the sample from configs/sample.toml and write it to data/sample.json. Run once; labels point at its answers.

    uv run python scripts/freeze_sample.py
"""

from llm_judge_vs_expert_labels.data.corpus import Corpus
from llm_judge_vs_expert_labels.data.sample import FROZEN, Sampler, freeze

if __name__ == "__main__":
    sampler = Sampler.from_config()
    sample = sampler.sample(Corpus.load().candidates())
    freeze(sample, sampler.split(sample))
    print(f"wrote {FROZEN}")
