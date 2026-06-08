from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
import numpy as np
import json
import time
import os
from .parsing import parse_arguments, parse_input_files
from .prompting import generate_prompt
from .decoding import (
    build_clean_vocab,
    get_allowed_tokens,
    get_allowed_ids_for_numbers,
    get_allowed_ids_for_strings,
    get_allowed_ids_for_booleans,
    apply_logits_mask,
)


def main() -> None:
    print("[*] Loading Model...")
    model = Small_LLM_Model()
    args = parse_arguments()
    list_of_functions, functions_tools, prompts_data, list_of_decode_name_functions = (
        parse_input_files(args, model)
    )
    clean_vocab = build_clean_vocab(model)
    parameters_injection = model.encode('", "parameters": {')[0].tolist()
    end_injection = model.encode("}")[0].tolist()

    final_results: list = []
    total_start = time.perf_counter()
    flag = True
    for item in prompts_data:
        gen = ""
        curr_key = ""
        state = "FUNCTION_NAME"
        schema_parameters: dict = {}

        raw_prompt_text = item["prompt"]
        print(f"[*] Processing: {raw_prompt_text}")

        safe_user_prompt = json.dumps(raw_prompt_text)
        prompt = generate_prompt(functions_tools, safe_user_prompt)
        tokens = model.encode(prompt)[0].tolist()

        while True:
            if state == "END":
                break
            if flag:
                logits = model.get_logits_from_input_ids(tokens)
            allowed_ids = []

            if state == "FUNCTION_NAME":
                allowed_ids = get_allowed_tokens(
                    list(list_of_functions.keys()), gen, clean_vocab
                )
            elif state == "PARAM_KEYS":
                for key in schema_parameters.keys():
                    tokens.extend(model.encode(f'"{key}": ')[0].tolist())
                    flag = False
                    curr_key = key
                    state = "PARAM_VALUES"
                    break
            elif state == "PARAM_VALUES":
                current_type = schema_parameters[curr_key]["type"]
                is_last_param = len(schema_parameters) == 1

                if current_type in ["number", "integer"]:
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

            if flag:
                masked_logits = apply_logits_mask(np.array(logits), allowed_ids)
                next_token = int(np.argmax(masked_logits))
                tokens.append(next_token)
                gen += clean_vocab[next_token]

            if state == "FUNCTION_NAME":
                reminders_of_tokens_of_function = []
                reminders_of_functions: list[int] = [
                    function_encode
                    for function_encode in list_of_decode_name_functions
                    if next_token in function_encode
                ]
                if len(reminders_of_functions) == 1:
                    index = reminders_of_functions[0].index(next_token)
                    reminders_of_tokens_of_function = reminders_of_functions[0][
                        index + 1 :
                    ]
                if gen in list_of_functions or reminders_of_tokens_of_function:
                    if reminders_of_tokens_of_function:
                        tokens.extend(reminders_of_tokens_of_function)
                        gen += model.decode(reminders_of_tokens_of_function)
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

            elif state == "PARAM_VALUES":
                current_type = schema_parameters[curr_key]["type"]
                is_value_complete = False
                flag = True

                if current_type in ["number", "boolean", "integer"]:
                    if "," in gen or "}" in gen:
                        is_value_complete = True
                    ending_part = gen
                elif current_type == "string":
                    clean_gen = gen.replace('\\"', "")
                    if clean_gen.count('"') >= 2 and (
                        "," in clean_gen.split('"')[-1]
                        or "}" in clean_gen.split('"')[-1]
                    ):
                        is_value_complete = True
                    ending_part = clean_gen.split('"')[-1]

                if is_value_complete:
                    if "," in gen:
                        del schema_parameters[curr_key]
                        gen = ""
                        state = "PARAM_KEYS" if schema_parameters else "END"
                    elif "}" in gen:
                        if ending_part.count("}") == 1:
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

    print("\n[+] Processing Complete!")
    print(
        "[+] Total execution time: ",
        f"{((time.perf_counter() - total_start) / 60):.2f} minutes",
    )
    print(f"[+] Results successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
