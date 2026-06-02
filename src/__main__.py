from llm_sdk import Small_LLM_Model
from pprint import pprint
import numpy as np
import json
import time
model = Small_LLM_Model()


# 
user = "What is the sum of 235 and 335?"
functions = [{"name": "fn_calculator_numbers", "description": "calculate two numbers are required together and type of opration are required and return their result.",
              "parameters": {"a": {"type": "number"}, "b": {"type": "number"}, "c": {"type": "string"}}, "returns": {"type": "number"}}]

prompt = 'You are AI assistant answer by using these functions tools:\n'
prompt += f"\n<tools>\n{functions[0]}\n</tools>"
prompt += '<|im_start|>following this template: {"prompt": <user-prompt>, "name": <function-name>, "arguments": <args-json-object>}<|im_end|>\n'
prompt += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prompt += "<|im_start|>If one of the parameters doesn't use it puts a default value based on it by the type of the parameter<|im_end|>"
prompt += f'{{"prompt": {user}, "name": '


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

tokens: list = model.encode(prompt)[0].tolist()
clean_vocab = build_clean_vocab(model)
with open(model.get_path_to_vocab_file()) as f:
    voc = f.read()

name_functions_allowed = [fun["name"] for fun in functions]

parameters = model.encode('", "parameters": {')[0].tolist()
gen = ""
curr_key = ""
state = "FUNCTION_NAME"

end = model.encode('}')[0].tolist()
schema_parameters = {}

json_result = ""
start = time.perf_counter()
flag = True
while True:
    if state == "END":
        break
    # if flag:
    logits: list = model.get_logits_from_input_ids(tokens)

    if state == "FUNCTION_NAME":
        allowed_ids = get_allowed_tokens(name_functions_allowed, gen, clean_vocab)

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
    # if flag:
    masked_logits = apply_logits_mask(logits, allowed_ids)
    next_token = int(np.argmax(masked_logits))
    tokens.append(next_token)
    gen += model.decode(next_token)
    print(gen)
    if state == "FUNCTION_NAME":
        if gen in name_functions_allowed:
            schema_parameters = functions[0]["parameters"]
            tokens.extend(parameters)
            if not schema_parameters:
                tokens.extend(end)
                state = "END"
            else:
                gen = ""
                state = "PARAM_KEYS"

    if state == "PARAM_KEYS":
        # for key in schema_parameters.keys():
        #     curr_key = key
        #     keys_encode = model.encode(f'"{key}": ')[0].tolist()
        #     tokens.extend(keys_encode)
        #     state = "PARAM_VALUES"
        #     gen = ""
        #     flag = True
        #     break
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
result = result[result.find(f'{{"prompt": {user}'):]
print(result.count("{") == result.count("}"))
print(result)
print(f"Execution time: {time.perf_counter() - start:.6f} seconds")