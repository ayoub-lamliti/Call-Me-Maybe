from llm_sdk import Small_LLM_Model
from pprint import pprint
import numpy as np
import json
import time
model = Small_LLM_Model()

functions = [{"name": "fn_add_numbers", "description": "Add three numbers are required together and return their sum.",
              "parameters": {"a": {"type": "number"}, "b": {"type": "number"}}, "returns": {"type": "number"}}]

prompt = 'What is the sum of 235 and 335? answer by using these function tools\t'
prompt += f'{functions[0]}\t'
prompt += '<think>\n\n</think>\t'
prompt += 'you are AI assistant answer by: {"prompt": <user-prompt>, "name": <function-name>, "arguments": <args-json-object>}\t'
prompt += '{"prompt": "What is the sum of 235 and 335?", "name": '


def get_allowed_tokens(tergets_list: list, logits: list,):
    allowed_ids = []
    for i, _ in enumerate(logits):
        text = gen + model.decode(i)
        for terget in tergets_list:
            if terget.startswith(text):
                allowed_ids.append(i)
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


def get_allowed_ids_for_numbers(clean_vocab: dict[int, str]):
    allowed_ids: list = []

    allowed_chars = set("0123456789.-,} ")
    for token_id, token_text in clean_vocab.items():
        if not token_text:
            continue
        if all(char in allowed_chars for char in token_text):
            allowed_ids.append(token_id)
    return allowed_ids


tokens: list = model.encode(prompt)[0].tolist()
clean_vocab = build_clean_vocab(model)
with open(model.get_path_to_vocab_file()) as f:
    voc = f.read()

name_functions_allowed = [fun["name"] for fun in functions]

pramas = [p for p in functions[0]["parameters"].keys()]

gen = ""
curr_key = ""
state = "FUNCTION_NAME"

parameters = model.encode('", "parameters": {')[0].tolist()
end = model.encode('}')[0].tolist()
schema_parameters = {}

json_result = '{"prompt": "What is the sum of 235 and 335?", "name": '
start = time.perf_counter()
while True:
    if state == "END":
        break
    logits: list = model.get_logits_from_input_ids(tokens)

    if state == "FUNCTION_NAME":
        allowed_ids = get_allowed_tokens(name_functions_allowed, logits)

    elif state == "PARAM_KEYS":
        keys = [f'"{key}" :' for key in schema_parameters.keys()]
        allowed_ids = get_allowed_tokens(keys, logits)

    elif state == "PARAM_VALUES":
        type_of_param = schema_parameters[curr_key]["type"]
        if type_of_param == "number":
            allowed_ids = get_allowed_ids_for_numbers(clean_vocab)

    masked_logits = apply_logits_mask(logits, allowed_ids)
    next_token = int(np.argmax(masked_logits))
    tokens.append(next_token)
    result = model.decode(next_token)
    gen += result
    json_result += result
    print(gen, flush=True)
    print("state :", state)
    
    if state == "FUNCTION_NAME":
        if gen in name_functions_allowed:
            tokens.extend(parameters)
            json_result += '", "parameters": {'
            schema_parameters = functions[0]["parameters"]
            # print(schema_parameters)
            if not schema_parameters:
                tokens.extend(end)
                state = "END"
            else:
                gen = ""
                state = "PARAM_KEYS"

    if state == "PARAM_KEYS":
        for key in schema_parameters.keys():
            curr_key = key
            keys_encode = model.encode(f'"{key}": ')[0].tolist()
            print(keys_encode, key)
            tokens.extend(keys_encode)
            state = "PARAM_VALUES"
            gen = ""
            break

    if state == "PARAM_VALUES":
        if "," in gen:
            del schema_parameters[curr_key]
            gen = ""
            state = "PARAM_KEYS" if schema_parameters else "END"
        elif "}" in gen:
            state = "END"
# comment test
result = model.decode(tokens)
print(f"Execution time: {time.perf_counter() - start:.6f} seconds")
print(json_result.count("{") == json_result.count("}"))
print(json_result)
print(result)