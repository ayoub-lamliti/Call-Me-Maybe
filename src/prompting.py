def generate_prompt(functions: str, user_prompt: str) -> str:
    """Build the full prompt string sent to the LLM for function-call
    generation.  The prompt primes the model to emit a JSON object whose
    `parameters` key holds the extracted arguments."""
    prompt = "You are AI assistant answer by using these functions tools:\n"
    prompt += f"<tools>\n{functions}\n</tools>\n"

    prompt += "CRITICAL INSTRUCTIONS:\n"
    prompt += "- Be extremely precise with strings and regex patterns.\n"
    prompt += (
        "- For regex matching letters (like vowels), ALWAYS"
        "include both uppercase and lowercase variations"
        "(e.g., [aeiouAEIOU]).\n"
    )
    prompt += (
        "- REPLACEMENT RULE: When replacing text with a symbol,"
        "NEVER use multiple characters. Always use a SINGLE character"
        "(e.g., '*' not '**'), even if the prompt uses plural words"
        "like 'asterisks'.\n"
    )
    prompt += (
        "- If a parameter is not used, provide a default value based"
        "on its type.\n"
    )

    prompt += (
        '<|im_start|>following this template: {"prompt": <user-prompt>,'
        '"name": <function-name>, "parameters": <args-json-object>}<|im_end|>\n'
    )
    prompt += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    prompt += f'{{"prompt": {user_prompt}, "name": "'
    return prompt
