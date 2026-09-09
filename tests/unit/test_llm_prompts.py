"""Tests for the LLM stage's prompt assembly and caching.

These run on CPU and load no model. The point is that everything expensive
happens on a rented GPU, so any bug that can be caught here must be caught here
— a prompt bug discovered after a full generation run costs real money.

Faithfulness targets come from `methods/flag/chat.py`, which is the authority
because the paper omits truncation, token limits and response parsing.
"""
from __future__ import annotations

import pathlib

from flagbench.llm.enhance import (
    ANSWER_SEPARATOR,
    MAX_NEW_TOKENS,
    TEXT_TRUNCATE_CHARS,
    LLMConfig,
    PromptSet,
    build_prompt,
    cache_key,
    parse_response,
    question_block,
    strip_numbering,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_prompts(dataset="reddit") -> PromptSet:
    return PromptSet.load(dataset, ROOT / "prompts")


# ------------------------------------------------------- upstream constants
def test_constants_match_upstream_literals():
    assert TEXT_TRUNCATE_CHARS == 1200      # chat.py:63
    assert MAX_NEW_TOKENS == 550            # chat.py:71
    assert ANSWER_SEPARATOR == "Answer:"    # chat.py:74


def test_default_config_uses_the_resolved_gemma_id():
    config = LLMConfig()
    assert config.model_id == "google/gemma-2-9b-it"
    assert config.dtype == "float16"
    assert config.max_new_tokens == 550
    assert config.truncate_chars == 1200


# ------------------------------------------------------------ prompt files
def test_all_four_prompts_load_for_both_datasets():
    for dataset in ("reddit", "instagram"):
        prompts = load_prompts(dataset)
        for text in (prompts.system_instruction, prompts.global_prompt,
                     prompts.discriminative, prompts.residual):
            assert text and text.strip()
        assert len(prompts.hashes) == 4


def test_reddit_and_instagram_prompts_genuinely_differ():
    """Not just noun swaps -- the discriminative prompts differ in substance."""
    reddit = load_prompts("reddit")
    instagram = load_prompts("instagram")
    assert reddit.discriminative != instagram.discriminative
    assert reddit.global_prompt != instagram.global_prompt
    # Reddit asks the model to relate text to classification...
    assert "directly relates to classifying" in reddit.discriminative
    # ...Instagram asks it NOT to predict the classification.
    assert "without predicting their classification" in instagram.discriminative


def test_system_instruction_is_shared_between_datasets():
    assert (
        load_prompts("reddit").hashes["system_instruction"]
        == load_prompts("instagram").hashes["system_instruction"]
    )


# ---------------------------------------------------------- question block
def test_question_block_numbers_and_brackets_each_text():
    block = question_block(["alpha", "beta"], noun="posts")
    assert block.startswith("The posts of these users are as follows:\n")
    assert "1. [alpha]" in block
    assert "2. [beta]" in block


def test_question_block_truncates_at_the_upstream_limit():
    long_text = "x" * 5000
    block = question_block([long_text], truncate=TEXT_TRUNCATE_CHARS)
    assert "x" * TEXT_TRUNCATE_CHARS in block
    assert "x" * (TEXT_TRUNCATE_CHARS + 1) not in block


def test_question_block_leaves_short_text_untouched():
    block = question_block(["short"], truncate=TEXT_TRUNCATE_CHARS)
    assert "1. [short]" in block


def test_question_block_noun_differs_per_dataset():
    """chat.py says 'posts', chat1.py says 'introductions'."""
    assert question_block(["a"], noun="posts").startswith("The posts")
    assert question_block(["a"], noun="introductions").startswith(
        "The introductions"
    )


# --------------------------------------------------------- prompt assembly
def test_prompt_layout_matches_upstream_ordering():
    prompts = load_prompts("reddit")
    prompt = build_prompt(prompts, "discriminative", ["a", "b"], LLMConfig())

    assert prompt.startswith(prompts.system_instruction)
    assert "\nPrompt: " in prompt
    assert prompts.global_prompt in prompt
    assert prompts.discriminative in prompt
    assert "\nQuestion: " in prompt
    assert prompt.rstrip().endswith(ANSWER_SEPARATOR)

    # Ordering: system -> global -> task -> question -> Answer:
    assert (
        prompt.index(prompts.global_prompt)
        < prompt.index(prompts.discriminative)
        < prompt.index("Question: ")
        < prompt.rindex(ANSWER_SEPARATOR)
    )


def test_discriminative_and_residual_produce_different_prompts():
    prompts = load_prompts("reddit")
    config = LLMConfig()
    disc = build_prompt(prompts, "discriminative", ["a"], config)
    res = build_prompt(prompts, "residual", ["a"], config)
    assert disc != res
    assert prompts.residual in res
    assert prompts.residual not in disc


def test_unknown_prompt_kind_is_rejected():
    try:
        load_prompts("reddit").prompt_for("nonsense")
    except ValueError:
        return
    raise AssertionError("unknown prompt kind should raise")


# -------------------------------------------------------- response parsing
def test_parse_response_splits_on_answer_and_newlines():
    answer = "preamble\nAnswer:\nfirst line\nsecond line\n"
    assert parse_response(answer, expected=2) == ["first line", "second line"]


def test_parse_response_rejects_a_wrong_line_count():
    """Upstream discards these; repairing would invent text for a node."""
    answer = "Answer:\nonly one line\n"
    assert parse_response(answer, expected=3) is None


def test_parse_response_ignores_blank_lines():
    answer = "Answer:\n\nfirst\n\n\nsecond\n\n"
    assert parse_response(answer, expected=2) == ["first", "second"]


def test_parse_response_without_the_separator_still_parses():
    assert parse_response("alpha\nbeta", expected=2) == ["alpha", "beta"]


# ------------------------------------------------------------- numbering
def test_strip_numbering_removes_enumeration_prefixes():
    assert strip_numbering(["1. alpha", "2. beta"]) == ["alpha", "beta"]
    assert strip_numbering(["10) gamma"]) == ["gamma"]


def test_strip_numbering_leaves_unnumbered_text_alone():
    assert strip_numbering(["no prefix here"]) == ["no prefix here"]


def test_strip_numbering_does_not_eat_leading_digits_of_content():
    """'2024 was a good year' must not lose its year."""
    assert strip_numbering(["2024 was a good year"]) == ["2024 was a good year"]


# ------------------------------------------------------------- cache keys
def test_cache_key_changes_with_every_meaningful_field():
    prompts = load_prompts("reddit")
    config = LLMConfig()
    base = cache_key("reddit", "sampA", prompts, config, "discriminative")

    assert base != cache_key("instagram", "sampA", prompts, config, "discriminative")
    assert base != cache_key("reddit", "sampB", prompts, config, "discriminative")
    assert base != cache_key("reddit", "sampA", prompts, config, "residual")
    assert base != cache_key(
        "reddit", "sampA", prompts,
        LLMConfig(max_new_tokens=100), "discriminative",
    )
    assert base != cache_key(
        "reddit", "sampA", prompts,
        LLMConfig(model_id="other/model"), "discriminative",
    )
    assert base == cache_key("reddit", "sampA", prompts, config, "discriminative")


def test_cache_key_changes_when_a_prompt_changes():
    """A cache hit across an edited prompt would mix two experiments."""
    prompts = load_prompts("reddit")
    config = LLMConfig()
    before = cache_key("reddit", "s", prompts, config, "discriminative")

    edited = PromptSet(
        dataset=prompts.dataset,
        system_instruction=prompts.system_instruction,
        global_prompt=prompts.global_prompt,
        discriminative=prompts.discriminative + " EDITED",
        residual=prompts.residual,
        hashes={**prompts.hashes, "discriminative": "different"},
    )
    assert cache_key("reddit", "s", edited, config, "discriminative") != before


def test_cache_key_embeds_a_readable_model_slug():
    prompts = load_prompts("reddit")
    key = cache_key("reddit", "s", prompts, LLMConfig(), "discriminative")
    assert "google_gemma-2-9b-it" in key
    assert key.startswith("reddit__discriminative__")


# --------------------------------------------------------- GPU-only policy
def test_enhancer_refuses_cpu_with_an_explanation():
    from flagbench.llm.enhance import LLMEnhancer

    enhancer = LLMEnhancer(LLMConfig(), device="cpu")
    try:
        enhancer.load()
    except RuntimeError as exc:
        message = str(exc)
        assert "GPU" in message and "D-003" in message
        assert "18.5 GB" in message
        return
    raise AssertionError("loading a 9B model on CPU should be refused")


def _main() -> int:
    tests = [
        (n, o) for n, o in sorted(globals().items())
        if n.startswith("test_") and callable(o)
    ]
    failed = []
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            failed.append(name)
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
