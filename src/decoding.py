import numpy as np
import json
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


def get_allowed_tokens(
    target_list: list[str], current_string: str, clean_vocab: dict[int, str]
) -> list[int]:
    allowed_ids: list[int] = []
    for token_id, vocab in clean_vocab.items():
        if not vocab:
            continue
        text = current_string + vocab
        for target in target_list:
            if target.startswith(text):
                allowed_ids.append(token_id)
                break
    return allowed_ids


def apply_logits_mask(logits: list, allowed_ids: list[int]) -> np.ndarray:
    masked_logits = np.full_like(logits, -np.inf)
    for token_id in allowed_ids:
        masked_logits[token_id] = logits[token_id]
    return masked_logits


def build_clean_vocab(model: Small_LLM_Model) -> dict[int, str]:
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r") as f:
        vocabulary = json.load(f)
    clean_vocab: dict[int, str] = {}
    for _, token_id in vocabulary.items():
        clean_text = model.decode([token_id])
        clean_vocab[token_id] = clean_text
    return clean_vocab


def get_allowed_ids_for_numbers(
    clean_vocab: dict[int, str], is_last: bool
) -> list[int]:
    allowed_ids: list[int] = []
    allowed_chars = set("0123456789.-} ") if is_last else set("0123456789.-, ")
    for token_id, token_text in clean_vocab.items():
        if (not token_text or token_text.count("}") > 1
                or token_text.count(",") > 1):
            continue
        if all(char in allowed_chars for char in token_text):
            allowed_ids.append(token_id)
    return allowed_ids


def get_allowed_ids_for_strings(
    clean_vocab: dict[int, str], is_last: bool
) -> list[int]:
    allowed_ids: list[int] = []
    for token_id, token_text in clean_vocab.items():
        if not token_text:
            continue
        if "\n" in token_text or "\r" in token_text:
            continue
        unescaped_text = token_text.replace('\\"', "")
        if '"' in unescaped_text:
            after_quote = unescaped_text[unescaped_text.find('"') + 1:]
            allowed_closing = "} " if is_last else ", "
            if any(char not in allowed_closing for char in after_quote):
                continue
            if after_quote.count("}") > 1 or after_quote.count(",") > 1:
                continue
        allowed_ids.append(token_id)
    return allowed_ids


def get_allowed_ids_for_booleans(
    clean_vocab: dict[int, str], gen: str, is_last: bool
) -> list[int]:
    if is_last:
        target_list = ["true}", "false}", "true }", "false }"]
    else:
        target_list = ["true,", "false,", "true ,", "false ,"]
    return get_allowed_tokens(target_list, gen, clean_vocab)
