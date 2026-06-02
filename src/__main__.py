from llm_sdk import Small_LLM_Model
from pprint import pprint
import numpy as np
import json
import time


model = Small_LLM_Model()
list_of_functions = {}


def generate_prompt(functions: str, user_prompt: str) -> str:
    prompt = 'You are AI assistant answer by using these functions tools:\n'
    prompt += f"<tools>\n{functions}\n</tools>\n"
    prompt += '<|im_start|>following this template: {"prompt": <user-prompt>, "name": <function-name>, "arguments": <args-json-object>}<|im_end|>\n'
    prompt += f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"
    prompt += "<|im_start|>If one of the parameters doesn't use it puts a default value based on it by the type of the parameter<|im_end|>\n"
    prompt += f'{{"prompt": {user_prompt}, "name": '
    return prompt


with (
    open("/home/alamliti/goinfre/Call/Call-Me-Maybe/data/input/functions_definition.json") as functions_definition,
    open("/home/alamliti/goinfre/Call/Call-Me-Maybe/data/input/function_calling_tests.json") as prompts,
    # open("/home/ayoub-lec/Documents/Call-Me-Maybe/data/input/function_calling_tests.json") as p,
    # open("/home/ayoub-lec/Documents/Call-Me-Maybe/data/input/functions_definition.json") as fun,
):
    objs = json.load(functions_definition)
    for obj in objs:
        list_of_functions[obj["name"]] = obj
    prompts: dict = json.load(prompts)
    functions_tools = "\n".join(json.dumps(line) for line in objs)



def get_allowed_tokens(tergets_list: list, current_string: str, clean_vocab: dict[int, str]):
    allowed_ids = []
    for token_id, vocab in clean_vocab.items():
        if not vocab:
            continue
        text = current_string + vocab
        for terget in tergets_list:
            if terget.startswith(text):
                allowed_ids.append(token_id)
                break
    return allowed_ids


def apply_logits_mask(logits, allowed_ids):
    masked_logits = np.full_like(logits, -np.inf)
    for token_id in allowed_ids:
        masked_logits[token_id] = logits[token_id]
    return masked_logits


def build_clean_vocab(model):
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path) as f:
        vocabulary = json.load(f)
    clean_vocab = {}
    for _, token_id in vocabulary.items():
        clean_text = model.decode([token_id])
        clean_vocab[token_id] = clean_text
    return clean_vocab


def get_allowed_ids_for_numbers(clean_vocab: dict[int, str]) -> list[int]:
    allowed_ids: list = []

    allowed_chars = set("0123456789.-,} ")
    for token_id, token_text in clean_vocab.items():
        if not token_text:
            continue
        if all(char in allowed_chars for char in token_text):
            allowed_ids.append(token_id)
    return allowed_ids

def get_allowed_ids_for_strings(clean_vocab: dict[int, str]) -> list[int]:
    allowed_ids: list[int] = []
    
    for token_id, token_text in clean_vocab.items():
        if not token_text:
            continue
        if "\n" in token_text or "\r" in token_text:
            continue    
        if '"' in token_text:
            after_quote = token_text[token_text.find('"') + 1:]
            if any(char not in ',} ' for char in after_quote):
                continue
        allowed_ids.append(token_id)
    return allowed_ids        

clean_vocab = build_clean_vocab(model)
parameters = model.encode('", "parameters": {')[0].tolist()
end = model.encode('}')[0].tolist()

print(list_of_functions)
def generate_json():
    for user_prompt in prompts:
        gen = ""
        curr_key = ""
        state = "FUNCTION_NAME"
        schema_parameters = {}
        start = time.perf_counter()
        prompt = generate_prompt(functions_tools, user_prompt["prompt"])
        tokens: list = model.encode(prompt)[0].tolist()

        while True:
            if state == "END":
                break
            logits: list = model.get_logits_from_input_ids(tokens)

            if state == "FUNCTION_NAME":
                allowed_ids = get_allowed_tokens(list_of_functions.keys(), gen, clean_vocab)

            elif state == "PARAM_KEYS":
                keys = [f'"{key}": ' for key in schema_parameters.keys()]
                allowed_ids = get_allowed_tokens(keys, gen, clean_vocab)

            elif state == "PARAM_VALUES":
                current_type = schema_parameters[curr_key]["type"]
                if current_type == "number":
                    allowed_ids = get_allowed_ids_for_numbers(clean_vocab)
                elif current_type == "string":
                    if gen == "":
                        allowed_ids = get_allowed_tokens(['"'], gen, clean_vocab)
                    else:
                        allowed_ids = get_allowed_ids_for_strings(clean_vocab)
            masked_logits = apply_logits_mask(logits, allowed_ids)
            next_token = int(np.argmax(masked_logits))
            tokens.append(next_token)
            gen += clean_vocab[next_token]
            if state == "FUNCTION_NAME":
                if gen in list_of_functions.keys():
                    schema_parameters = list_of_functions[gen]["parameters"]
                    tokens.extend(parameters)
                    if not schema_parameters:
                        tokens.extend(end)
                        state = "END"
                    else:
                        gen = ""
                        state = "PARAM_KEYS"

            if state == "PARAM_KEYS":
                target_keys = [f'"{key}": ' for key in schema_parameters.keys()]
                if gen in target_keys:
                    curr_key = gen.split('"')[1]
                    gen = ""
                    state = "PARAM_VALUES"

            elif state == "PARAM_VALUES":
                current_type = schema_parameters[curr_key]["type"]
                is_value_complete = False
                if current_type == "number":
                    if "," in gen or "}" in gen:
                        is_value_complete = True
                elif current_type == "string":
                    if gen.count('"') >= 2 and ("," in gen.split('"')[-1] or "}" in gen.split('"')[-1]):
                        is_value_complete = True
                if is_value_complete:
                    if "," in gen:
                        del schema_parameters[curr_key]
                        gen = ""
                        state = "PARAM_KEYS" if schema_parameters else "END"
                    elif "}" in gen:
                        state = "END"

        result = model.decode(tokens)
        print(f"Execution time: {time.perf_counter() - start:.6f} seconds")
        result = result[result.find(f'{{"prompt": {user_prompt}'):]
        print(result.count("{") == result.count("}"))
        print(result)

generate_json()