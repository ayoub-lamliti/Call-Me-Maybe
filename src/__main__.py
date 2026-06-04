from llm_sdk import Small_LLM_Model
import numpy as np
import argparse
import json
import time
import sys
import os


def generate_prompt(functions: str, user_prompt: str) -> str:
    prompt = "You are AI assistant answer by using these functions tools:\n"
    prompt += f"<tools>\n{functions}\n</tools>\n"
    prompt += '<|im_start|>following this template: {"prompt": <user-prompt>, "name": <function-name>, "arguments": <args-json-object>}<|im_end|>\n'
    prompt += "<|im_start|>If one of the parameters doesn't use it puts a default value based on it by the type of the parameter<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
    prompt += "\n<|im_start|>assistant\n"
    prompt += f'{{"prompt": {user_prompt}, "name": "'
    return prompt

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
    allowed_chars = set("0123456789.-} ") if is_last else set("0123456789.-, ")
    for token_id, token_text in clean_vocab.items():
        if not token_text or token_text.count("}") > 1 or token_text.count(",") > 1:
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
            allowed_closing = "} " if is_last else ", "
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


def main():

    parser = argparse.ArgumentParser(description="LLM Function Calling Inference")
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to functions definition JSON",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to prompts JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Path to save the output JSON",
    )
    args = parser.parse_args()

    try:
        with open(args.functions_definition, "r") as f:
            objs = json.load(f)
            list_of_functions = {obj["name"]: obj for obj in objs}
            functions_tools = "\n".join(json.dumps(line) for line in objs)

        with open(args.input, "r") as f:
            prompts_data = json.load(f)
    except Exception as e:
        print(f"[-] Error loading input files: {e}")
        sys.exit(1)

    print("[*] Loading Model...")
    model = Small_LLM_Model()
    clean_vocab = build_clean_vocab(model)
    parameters_injection = model.encode('", "parameters": {')[0].tolist()
    end_injection = model.encode("}")[0].tolist()

    final_results = []
    total_start = time.perf_counter()

    for item in prompts_data:
        gen = ""
        curr_key = ""
        state = "FUNCTION_NAME"
        schema_parameters = {}

        raw_prompt_text = item["prompt"]
        print(f"[*] Processing: {raw_prompt_text}")

        safe_user_prompt = json.dumps(raw_prompt_text)
        prompt = generate_prompt(functions_tools, safe_user_prompt)
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
                    allowed_ids = get_allowed_ids_for_booleans(
                        clean_vocab, gen, is_last_param
                    )

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
                    clean_gen = gen.replace('\\"', "")
                    if clean_gen.count('"') >= 2 and (
                        "," in clean_gen.split('"')[-1]
                        or "}" in clean_gen.split('"')[-1]
                    ):
                        is_value_complete = True

                if is_value_complete:
                    if "," in gen:
                        del schema_parameters[curr_key]
                        gen = ""
                        state = "PARAM_KEYS" if schema_parameters else "END"
                    elif "}" in gen:
                        tokens.extend(end_injection)
                        state = "END"

        result_raw = model.decode(tokens)

        try:
            json_start_index = result_raw.find(f'{{"prompt": {safe_user_prompt}')
            if json_start_index == -1:
                raise ValueError("JSON start object not found.")

            clean_json_str = result_raw[json_start_index:]
            parsed_json = json.loads(clean_json_str)
            func_name = parsed_json.get("name")
            if func_name in list_of_functions:
                original_schema = list_of_functions[func_name].get("parameters", {})

                for param_key, param_value in parsed_json.get("parameters", {}).items():
                    if original_schema.get(param_key, {}).get("type") == "number":
                        parsed_json["parameters"][param_key] = float(param_value)
            final_results.append(parsed_json)

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[-] CRITICAL ERROR on prompt: {raw_prompt_text}")
            print(f"Error details: {e}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"\n[+] Processing Complete!")
    print(f"[+] Total execution time: {((time.perf_counter() - total_start) / 60):.2f} minutes")
    print(f"[+] Results successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
