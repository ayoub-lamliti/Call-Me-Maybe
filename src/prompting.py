def generate_prompt(functions: str, user_prompt: str) -> str:
    """Build the full prompt string sent to the LLM for function-call
    generation.  The prompt primes the model to emit a JSON object whose
    `parameters` key holds the extracted arguments."""

    prompt = f"Tools:\n{functions}\n"
    prompt += "- Be extremely precise with strings and regex patterns.\n"
    prompt += 'Example:\n{"name":"function-name","parameters":<arguments>}\n'
    prompt += "Assistant:\n"
    prompt += f"User: {user_prompt}\n"
    prompt += '{"name":"'
    return prompt
