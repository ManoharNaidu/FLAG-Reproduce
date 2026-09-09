"""LLM text enhancement: the GPU-only stage of the FLAG pipeline.

Decision D-003 makes this the ONLY component that requires a GPU, and the
maintainer runs it on rented vast.ai instances. Two consequences shape the design:

1. **It must be severable.** This stage consumes a self-contained bundle (node
   texts + subgraph definitions + prompts) and emits a self-contained,
   checksummed text cache. It never needs the graph structure, the labels, the
   splits or the training loop. So a rented instance is used *only* for
   generation, and every downstream experiment replays locally from cache at no
   GPU cost. That also satisfies Phase 33's LLM cost control by construction.

2. **The cache must be unambiguous.** It is keyed by dataset version, sampling
   config, prompt version, model id and decoding parameters. A cache hit can
   never silently cross a prompt or model change, because that would mix two
   different experiments under one name.

Faithfulness to upstream (`methods/flag/chat.py`), which is the authority here
since the paper omits these details:

    * one prompt per SUBGRAPH, listing every node's text as a numbered item
    * per-node truncation at 1200 characters
    * max_new_tokens = 550
    * the answer is split on 'Answer:' and then on newlines
    * a response whose line count != the subgraph size is DISCARDED, not repaired

That last rule matters. Upstream keeps the batch but attaches no text, so those
nodes silently fall back to raw embeddings. We record the failure explicitly
instead, so the fraction of nodes that actually received LLM text is visible in
the manifest rather than being invisible.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import pathlib
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

ROOT = pathlib.Path(__file__).resolve().parents[3]
PROMPTS_DIR = ROOT / "prompts"

# Upstream literals (methods/flag/chat.py, verified by test).
TEXT_TRUNCATE_CHARS = 1200
MAX_NEW_TOKENS = 550
ANSWER_SEPARATOR = "Answer:"


@dataclass(frozen=True)
class LLMConfig:
    """Decoding configuration. Every field enters the cache key."""

    model_id: str = "google/gemma-2-9b-it"
    """The paper writes 'Gemma-9b-it', which is not a valid HuggingFace id.
    Upstream hardcodes the bare directory name 'gemma-2-9b-it'. This is the
    resolved id -- see research/paper_notes.md section 3.1."""

    dtype: str = "float16"          # upstream: torch_dtype=torch.float16
    max_new_tokens: int = MAX_NEW_TOKENS
    truncate_chars: int = TEXT_TRUNCATE_CHARS
    do_sample: bool = False
    """Upstream calls `generate()` with no sampling arguments, so it uses the
    model's generation defaults. We pin greedy decoding explicitly: sampling
    without a fixed seed would make the cache non-reproducible, which is worse
    than a small deviation from an unstated default. Recorded as a deviation."""
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int = 0
    batch_prompts: int = 1
    """Subgraph prompts per forward pass. Upstream does 1. >1 needs left padding
    and changes nothing semantically, but is recorded anyway."""

    def as_record(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class PromptSet:
    """The four verbatim prompts for one dataset, plus their hashes."""

    dataset: str
    system_instruction: str
    global_prompt: str
    discriminative: str
    residual: str
    version: str = "flag-official-v1"
    hashes: dict = field(default_factory=dict)

    @classmethod
    def load(cls, dataset: str, prompts_dir: pathlib.Path | None = None) -> PromptSet:
        directory = pathlib.Path(prompts_dir or PROMPTS_DIR) / dataset
        if not directory.exists():
            raise FileNotFoundError(
                f"{directory} missing. Run:\n"
                f"  python -m scripts.preprocess.extract_prompts"
            )
        parts, hashes = {}, {}
        for role in ("system_instruction", "global", "discriminative", "residual"):
            path = directory / f"{role}.txt"
            text = path.read_text(encoding="utf-8")
            parts[role] = text
            hashes[role] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return cls(
            dataset=dataset,
            system_instruction=parts["system_instruction"],
            global_prompt=parts["global"],
            discriminative=parts["discriminative"],
            residual=parts["residual"],
            hashes=hashes,
        )

    def prompt_for(self, kind: str) -> str:
        if kind == "discriminative":
            return self.discriminative
        if kind == "residual":
            return self.residual
        raise ValueError(f"unknown prompt kind {kind!r}")


def question_block(texts: list[str], truncate: int = TEXT_TRUNCATE_CHARS,
                   noun: str = "posts") -> str:
    """Numbered list of node texts, exactly as upstream builds it.

    Upstream (`chat.py:61-66`):
        question = "The posts of these users are as follows:\\n"
        for i in range(len(batch.subset)):
            if len(text) > 1200: question += f"{i+1}. [{text[:1200]}]\\n"
            else:                question += f"{i+1}. [{text}]\\n"
    """
    lines = [f"The {noun} of these users are as follows:"]
    for index, text in enumerate(texts, start=1):
        clipped = text[:truncate] if len(text) > truncate else text
        lines.append(f"{index}. [{clipped}]")
    return "\n".join(lines) + "\n"


def build_prompt(prompts: PromptSet, kind: str, texts: list[str],
                 config: LLMConfig, noun: str = "posts") -> str:
    """Assemble the full prompt string, matching upstream's layout exactly.

    Upstream (`chat.py:67`):
        f"{system_instruction}\\nPrompt: {global_prompt}\\n{unique_prompt}\\n"
        f"Question: {question}\\nAnswer:"
    """
    question = question_block(texts, config.truncate_chars, noun)
    return (
        f"{prompts.system_instruction}\n"
        f"Prompt: {prompts.global_prompt}\n"
        f"{prompts.prompt_for(kind)}\n"
        f"Question: {question}\n"
        f"{ANSWER_SEPARATOR}"
    )


def parse_response(answer: str, expected: int) -> list[str] | None:
    """Split the model's answer into one line per node.

    Returns None when the line count does not match, which upstream treats as a
    failure for that subgraph. We keep that behaviour -- repairing a malformed
    response would silently invent text for a node.
    """
    if ANSWER_SEPARATOR in answer:
        answer = answer.split(ANSWER_SEPARATOR, 1)[1]
    lines = [line.strip() for line in answer.split("\n") if line.strip()]
    if len(lines) != expected:
        return None
    return lines


def strip_numbering(lines: list[str]) -> list[str]:
    """Remove a leading '1. ' / '12. ' enumeration if the model echoed it.

    Upstream does this with fixed slicing -- `text[i][2:]` for train but
    `text[i][3:]` for val/test (`encode.py:38,51,63`), an off-by-one that trims
    the two splits differently (flag_code_audit.md 5.7). We strip the actual
    prefix by pattern instead, which is what the fixed slices were approximating,
    and record the deviation.
    """
    import re

    pattern = re.compile(r"^\s*\d+\s*[.)]\s*")
    return [pattern.sub("", line).strip() for line in lines]


def cache_key(dataset: str, sampling_key: str, prompts: PromptSet,
              config: LLMConfig, kind: str) -> str:
    """Stable identity for one generation run.

    Includes the prompt hashes, so editing a prompt invalidates the cache
    instead of silently reusing text generated from a different instruction.
    """
    payload = json.dumps(
        {
            "dataset": dataset,
            "sampling": sampling_key,
            "kind": kind,
            "prompt_version": prompts.version,
            "prompt_hashes": prompts.hashes,
            "llm": config.as_record(),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    model_slug = config.model_id.replace("/", "_")
    return f"{dataset}__{kind}__{model_slug}__{sampling_key}__{digest}"


@dataclass
class GenerationStats:
    total_subgraphs: int = 0
    succeeded: int = 0
    format_failures: int = 0
    total_nodes: int = 0
    nodes_with_text: int = 0
    seconds: float = 0.0

    def as_record(self) -> dict:
        record = dataclasses.asdict(self)
        record["subgraph_success_rate"] = (
            round(self.succeeded / self.total_subgraphs, 4)
            if self.total_subgraphs else 0.0
        )
        record["node_coverage"] = (
            round(self.nodes_with_text / self.total_nodes, 4)
            if self.total_nodes else 0.0
        )
        return record


class LLMEnhancer:
    """Generates discriminative / residual text for sampled subgraphs.

    The model is loaded lazily so that this module can be imported, tested and
    its prompts inspected on a CPU-only machine without pulling 18.5 GB of
    weights.
    """

    def __init__(self, config: LLMConfig, device: str = "cuda:0"):
        self.config = config
        self.device = device
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not str(self.device).startswith("cuda"):
            raise RuntimeError(
                f"{self.config.model_id} needs a GPU (decision D-003). "
                f"In float16 the weights alone are about 18.5 GB, plus a KV "
                f"cache for {self.config.max_new_tokens} new tokens, and fp16 "
                f"is not meaningfully supported on CPU. Requested device: "
                f"{self.device!r}. No smaller substitute model is provided, "
                f"because its numbers would not be comparable to the paper's."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Run this stage on a GPU instance; see "
                "docs/vastai_gpu_workflow.md."
            )

        dtype = getattr(torch, self.config.dtype)
        logger.info("loading %s (%s)", self.config.model_id, self.config.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id, torch_dtype=dtype
        ).to(self.device)
        self._model.eval()

    def generate_one(self, prompt: str) -> str:
        import torch

        self.load()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature if self.config.do_sample else None,
                top_p=self.config.top_p if self.config.do_sample else None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        return self._tokenizer.decode(output[0], skip_special_tokens=True)

    def enhance(
        self,
        subgraphs,
        raw_texts: list[str],
        prompts: PromptSet,
        kind: str,
        noun: str = "posts",
        progress: bool = True,
    ) -> tuple[dict[int, list[str]], GenerationStats]:
        """Generate text for every subgraph.

        Returns `({central_node_id: [text per node in subset]}, stats)`.
        A subgraph whose response is malformed is omitted from the mapping, and
        counted in `stats.format_failures`.
        """
        stats = GenerationStats(total_subgraphs=len(subgraphs))
        results: dict[int, list[str]] = {}

        iterator = subgraphs
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(subgraphs, desc=f"LLM {kind}")
            except ImportError:
                pass

        start = time.time()
        for subgraph in iterator:
            node_ids = subgraph.subset.tolist()
            texts = [raw_texts[i] for i in node_ids]
            stats.total_nodes += len(node_ids)

            prompt = build_prompt(prompts, kind, texts, self.config, noun)
            answer = self.generate_one(prompt)
            lines = parse_response(answer, expected=len(node_ids))

            if lines is None:
                stats.format_failures += 1
                continue
            results[subgraph.central] = strip_numbering(lines)
            stats.succeeded += 1
            stats.nodes_with_text += len(node_ids)

        stats.seconds = round(time.time() - start, 1)
        return results, stats
