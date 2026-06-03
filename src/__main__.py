from llm_sdk import Small_LLM_Model
from pprint import pprint
import numpy as np
import json
import time

model = Small_LLM_Model()
list_of_functions = {}


def generate_prompt(functions: str, user_prompt: str) -> str:
    prompt = "You are AI assistant answer by using these functions tools:\n"
    prompt += f"<tools>\n{functions}\n</tools>\n"
    prompt += '<|im_start|>following this template: {"prompt": <user-prompt>, "name": <function-name>, "arguments": <args-json-object>}<|im_end|>\n'
    prompt += f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
    prompt += "<|im_start|>If one of the parameters doesn't use it puts a default value based on it by the type of the parameter<|im_end|>\n"
    prompt += f'{{"prompt": {user_prompt}, "name": "'
    return prompt


with (
    open(
        "/home/alamliti/goinfre/Call/Call-Me-Maybe/data/input/functions_definition.json"
        # "/home/ayoub-lec/Documents/test-call/data/input/functions_definition.json"
    ) as functions_definition,
    open(
        "/home/alamliti/goinfre/Call/Call-Me-Maybe/data/input/function_calling_tests.json"
        # "/home/ayoub-lec/Documents/test-call/data/input/function_calling_tests.json"
    ) as prompts,
):
    objs = json.load(functions_definition)
    for obj in objs:
        list_of_functions[obj["name"]] = obj
    prompts_data: dict = json.load(prompts)
    functions_tools = "\n".join(json.dumps(line) for line in objs)


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


def apply_logits_mask(logits, allowed_ids: list[int]):
    masked_logits = np.full_like(logits, -np.inf)
    for token_id in allowed_ids:
        masked_logits[token_id] = logits[token_id]
    return masked_logits


def build_clean_vocab(model) -> dict[int, str]:
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
    allowed_chars = set("0123456789.-} ") if is_last else set("0123456789.-,} ")

    for token_id, token_text in clean_vocab.items():
        if not token_text:
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
            after_quote = unescaped_text[unescaped_text.find('"') + 1 :]
            allowed_closing = "} " if is_last else ",} "
            if any(char not in allowed_closing for char in after_quote):
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


clean_vocab = build_clean_vocab(model)
parameters_injection = model.encode('", "parameters": {')[0].tolist()
end_injection = model.encode("}")[0].tolist()


def generate_json():
    for user_prompt in prompts_data:
        gen = ""
        curr_key = ""
        state = "FUNCTION_NAME"
        schema_parameters = {}
        start = time.perf_counter()

        user_prompt = json.dumps(user_prompt["prompt"])
        prompt = generate_prompt(functions_tools, user_prompt)
        tokens: list = model.encode(prompt)[0].tolist()

        while True:
            if state == "END":
                break

            logits: list = model.get_logits_from_input_ids(tokens)
            allowed_ids = []

            if state == "FUNCTION_NAME":
                allowed_ids = get_allowed_tokens(
                    list(list_of_functions.keys()), gen, clean_vocab
                )

            elif state == "PARAM_KEYS":
                keys = [f'"{key}": ' for key in schema_parameters.keys()]
                allowed_ids = get_allowed_tokens(keys, gen, clean_vocab)

            elif state == "PARAM_VALUES":
                current_type = schema_parameters[curr_key]["type"]
                is_last_param = len(schema_parameters) == 1

                if current_type == "number":
                    allowed_ids = get_allowed_ids_for_numbers(
                        clean_vocab, is_last_param
                    )
                elif current_type == "string":
                    if gen == "":
                        allowed_ids = get_allowed_tokens(['"'], gen, clean_vocab)
                    else:
                        allowed_ids = get_allowed_ids_for_strings(
                            clean_vocab, is_last_param
                        )
                elif current_type == "boolean":
                    allowed_ids = get_allowed_ids_for_booleans(clean_vocab, gen, is_last_param)

            masked_logits = apply_logits_mask(np.array(logits), allowed_ids)
            next_token = int(np.argmax(masked_logits))
            tokens.append(next_token)
            gen += clean_vocab[next_token]

            if state == "FUNCTION_NAME":
                if gen in list_of_functions:
                    schema_parameters = (
                        list_of_functions[gen].get("parameters", {}).copy()
                    )
                    tokens.extend(parameters_injection)

                    if not schema_parameters:
                        tokens.extend(end_injection)
                        state = "END"
                    else:
                        gen = ""
                        state = "PARAM_KEYS"

            elif state == "PARAM_KEYS":
                target_keys = [f'"{key}": ' for key in schema_parameters.keys()]
                if gen in target_keys:
                    curr_key = gen.split('"')[1]
                    gen = ""
                    state = "PARAM_VALUES"

            elif state == "PARAM_VALUES":
                current_type = schema_parameters[curr_key]["type"]
                is_value_complete = False

                if current_type in ["number", "boolean"]:
                    if "," in gen or "}" in gen:
                        is_value_complete = True

                elif current_type == "string":
                    clean_gen = gen.replace('\\"', '')
                    if clean_gen.count('"') >= 2 and (
                        "," in clean_gen.split('"')[-1] or "}" in clean_gen.split('"')[-1]
                    ):
                        is_value_complete = True

                if is_value_complete:
                    if "," in gen:
                        del schema_parameters[curr_key]
                        gen = ""
                        state = "PARAM_KEYS" if schema_parameters else "END"
                    elif "}" in gen:
                        state = "END"

        print(f"Execution time: {time.perf_counter() - start:.6f} seconds")
        result_raw = model.decode(tokens)

        try:
            json_start_index = result_raw.find(f'{{"prompt": {user_prompt}')
            if json_start_index == -1:
                raise ValueError("JSON start object not found.")

            clean_json_str = result_raw[json_start_index:]
            parsed_json = json.loads(clean_json_str)

            print("[+] VALID JSON GENERATED:")
            pprint(parsed_json)

        except (json.JSONDecodeError, ValueError) as e:
            print("[-] CRITICAL ERROR: Invalid JSON Generated!")
            print(f"Error details: {e}")
        print("-" * 50)


if __name__ == "__main__":
    start = time.perf_counter()
    generate_json()
    print(f"Total of execution time: {time.perf_counter() - start:.6f} seconds")
