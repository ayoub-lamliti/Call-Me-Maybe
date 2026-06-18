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
    build_prefix_cache,
)
from .banner import Fore, banner


def main() -> None:
    """Entry point: load model, process all prompts, write results."""
    print(Fore.GREEN + banner)
    print("[*] Loading Model...")
    args = parse_arguments()
    model = Small_LLM_Model(args.model)
    (
        list_of_functions,
        functions_tools,
        prompts_data,
    ) = parse_input_files(args, model)
    clean_vocab = build_clean_vocab(model)
    print("[*] Building Prefix Cache for Function Names...")
    prefix_cache = build_prefix_cache(
        list(list_of_functions.keys()), clean_vocab)
    parameters_injection = model.encode('","parameters":{')[0].tolist()
    end_injection = model.encode("}")[0].tolist()
    numbers = get_allowed_ids_for_numbers(clean_vocab, False)
    numbers_in_case_is_last = get_allowed_ids_for_numbers(clean_vocab, True)
    strings = get_allowed_ids_for_strings(clean_vocab, False)
    strings_in_case_is_last = get_allowed_ids_for_strings(clean_vocab, True)
    final_results: list = []
    total_start = time.perf_counter()
    for item in prompts_data:
        gen = ""
        curr_key = ""
        state = "FUNCTION_NAME"
        schema_parameters: dict = {}
        flag = True
        raw_prompt_text = item["prompt"]
        name_function = ""
        safe_user_prompt = json.dumps(raw_prompt_text)
        print(f"[*] Processing: {safe_user_prompt}")
        prompt = generate_prompt(functions_tools, safe_user_prompt)
        tokens = model.encode(prompt)[0].tolist()
        while state != "END":
            allowed_ids = []
            if flag:
                logits = model.get_logits_from_input_ids(tokens)
            if state == "FUNCTION_NAME":
                allowed_ids = prefix_cache.get(gen, [])
            elif state == "PARAM_KEYS":
                for key in schema_parameters.keys():
                    tokens.extend(model.encode(f'"{key}":')[0].tolist())
                    flag = False
                    curr_key = key
                    state = "PARAM_VALUES"
                    break
            elif state == "PARAM_VALUES":
                current_type = schema_parameters[curr_key]["type"]
                is_last_param = len(schema_parameters) == 1

                if current_type in ["number", "integer"]:
                    allowed_ids = (
                        numbers_in_case_is_last if
                        is_last_param else numbers
                    )
                elif current_type == "string":
                    if gen == "":
                        allowed_ids = get_allowed_tokens(
                            ['"'],
                            gen,
                            clean_vocab,
                        )
                    else:
                        allowed_ids = (
                            strings_in_case_is_last if
                            is_last_param else strings
                        )
                elif current_type == "boolean":
                    allowed_ids = get_allowed_ids_for_booleans(
                        clean_vocab, gen, is_last_param
                    )
            next_token: int = -1
            if flag:
                masked_logits = apply_logits_mask(
                    np.array(logits), allowed_ids)
                next_token = int(np.argmax(masked_logits))
                tokens.append(next_token)
                gen += clean_vocab[next_token]

            if state == "FUNCTION_NAME":
                if gen in list_of_functions:
                    name_function = gen
                    schema_parameters = list_of_functions[gen].get(
                        "parameters", {}).copy()
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
                    ending_part = clean_gen.split('"')[-1]
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
                        if ending_part.count("}") == 1:
                            tokens.extend(end_injection)
                        state = "END"
        result_raw = model.decode(tokens)

        try:
            json_start_index = result_raw.find(
                f'"name":"{name_function}"')
            if json_start_index == -1:
                raise ValueError("JSON start object not found.")

            clean_json_str = f'{{"prompt":{safe_user_prompt},'
            clean_json_str += result_raw[json_start_index:]
            parsed_json = json.loads(clean_json_str)
            func_name = parsed_json.get("name")
            if func_name in list_of_functions:
                original_schema = list_of_functions[func_name].get(
                    "parameters", {})

            for param_key, param_value in parsed_json.get(
                    "parameters", {}).items():
                expected_type = original_schema.get(param_key, {}).get(
                    "type")
                try:
                    if expected_type == "number":
                        parsed_json["parameters"][param_key] = float(
                            param_value)
                    elif expected_type == "integer":
                        parsed_json["parameters"][param_key] = int(param_value)
                except (ValueError, TypeError):
                    pass
            final_results.append(parsed_json)

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[-] CRITICAL ERROR on prompt: {raw_prompt_text}")
            print(f"Error details: {e}")
            final_results.append(
                {"prompt": raw_prompt_text, "name": None, "parameters": {}}
            )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(final_results, f, indent=4)

    print("\n[+] Processing Complete!")
    print(
        "[+] Total execution time: ",
        f"{((time.perf_counter() - total_start) / 60):.2f} minutes",
    )
    print(f"[+] Results successfully saved to: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        ...
