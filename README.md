*This project has been created as part of the 42 curriculum by alamliti.

# call me maybe

## Description

A function calling tool that translates natural language prompts into structured function calls using a small language model (Qwen3-0.6B) with constrained decoding. Given a prompt like "What is the sum of 40 and 2?", the system does not answer the question — it identifies the right function to call and extracts its arguments in valid JSON format.

The core challenge is reliability: a 0.6B model left to its own devices produces valid JSON roughly 30% of the time. This project forces it to 100% by implementing constrained decoding — intervening at the token level to mask any token that would violate the expected JSON schema before selection happens.

## Instructions

**Requirements:** Python 3.10+, [`uv`](https://github.com/astral-sh/uv)

```bash
# Install dependencies
make install

# Run
make run

# Or with explicit paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json

# Lint
make lint

# Debug
make debug

# Clean
make clean
```

Place the `llm_sdk/` directory at the same level as `src/`. The reviewer runs `uv sync` — no manual installs needed.

## Algorithm

The generation pipeline at a high level:

1. **Function selection** — The LLM is prompted with the available function definitions and the user query. Constrained decoding restricts the output to valid function names only; any token that would not continue a known function name string gets its logit set to `-inf` before sampling.

2. **Argument extraction** — Once the function is selected, its schema is known. The decoder then generates the JSON arguments field-by-field, restricting tokens at each position to only those that maintain a valid partial JSON object matching the expected parameter names and types.

3. **Schema enforcement** — Type constraints (number, string, boolean) are enforced per-field. The vocabulary JSON provided by the SDK is used to map token IDs to their string representations, allowing the decoder to reason about which tokens extend a valid partial output.

The result is always parseable, always schema-compliant — not because the model learned to behave, but because invalid outputs are structurally impossible.

## Design Decisions

- **Pydantic for all models** — function definitions, output records, and intermediate state are all validated with Pydantic, catching type mismatches early.
- **No DSPy, no transformers, no outlines** — constrained decoding is implemented from scratch against the `llm_sdk` interface.
- **Greedy selection after masking** — after applying the logit mask, the highest-probability valid token is selected. No temperature, no sampling noise needed for structured output.
- **Stateless per prompt** — each prompt is processed independently; no shared state between calls.

## Performance

| Metric | Target | Result |
|---|---|---|
| JSON validity | 100% | 100% (guaranteed by construction) |
| Function selection accuracy | 90%+ | — |
| Throughput | < 5 min for full test set | — |

Fill in the result column after running your own benchmarks.

## Challenges

- **Token boundary mismatches** — JSON strings like `"fn_add_numbers"` may be split across multiple tokens in unexpected ways. The decoder must track partial string matches across the vocabulary rather than doing simple string equality.
- **Number tokenization** — numbers like `265` or `3.14` can tokenize as one token or several. The constraint logic has to handle both cases while maintaining valid partial JSON.
- **Ambiguous prompts** — when a prompt could map to more than one function, the LLM's probability distribution over function names is the only signal. No heuristics allowed.

## Testing Strategy

- Unit tests (pytest/unittest) cover the constrained decoder independently of the LLM — given a fixed vocabulary and a partial JSON state, verify that only valid token IDs pass the mask.
- Integration tests run the full pipeline on the provided sample inputs and assert the output is valid JSON matching the schema.
- Edge cases tested: empty strings, large numbers, special characters, multi-parameter functions, ambiguous prompts.

## Example Usage

```bash
$ uv run python -m src
# reads from data/input/ by default, writes to data/output/

$ cat data/output/function_calls.json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

## Resources

- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Constrained decoding / structured generation overview](https://lmsys.org/blog/2024-02-05-compressed-fsm/)
- [JSON schema specification](https://json-schema.org/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)

**AI usage:** Claude was used to generate the initial README draft and to discuss constrained decoding implementation strategies. All code was written, reviewed, and understood by the authors. No AI-generated code was submitted without full comprehension and manual verification.