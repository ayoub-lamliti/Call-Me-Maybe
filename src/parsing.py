from pydantic import BaseModel, ValidationError
from typing import Any
import argparse
import json
import sys
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class FunctionParameter(BaseModel):
    """Schema for a single function parameter."""
    type: str


class FunctionDefinition(BaseModel):
    """Schema for a complete function definition."""
    name: str
    description: str = ""
    parameters: dict[str, FunctionParameter] = {}
    returns: dict[str, Any] = {}


class Prompt(BaseModel):
    """Schema for a single test prompt."""
    prompt: str


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments and return the namespace."""
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing function definitions.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing test prompts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path where the output JSON results will be written.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="The Hugging Face repository ID (default: %(default)s).",
    )
    return parser.parse_args()


def parse_input_files(
    args: argparse.Namespace, model: Small_LLM_Model
) -> tuple[dict, str, list, list]:
    """Load and validate both input files; encode function names.

    Returns:
        list_of_functions: dict mapping function name → validated dump.
        functions_tools: newline-joined JSON strings for the prompt.
        validated_prompts: list of prompt dicts.
        list_of_decode_name_functions: token-ID lists for each function name.
    """
    try:
        with open(args.functions_definition, "r") as f:
            raw_funcs = json.load(f)
        validated_funcs = [FunctionDefinition(**obj) for obj in raw_funcs]
        list_of_functions = {func.name: func.model_dump()
                             for func in validated_funcs}
        functions_tools = "\n".join(
            f'{func.name}({", ".join(func.parameters.keys())}):{func.description}'
            for func in validated_funcs
        )
        with open(args.input, "r") as f:
            raw_prompts = json.load(f)
        validated_prompts = [Prompt(**obj).model_dump() for obj in raw_prompts]
        return (
            list_of_functions,
            functions_tools,
            validated_prompts,
        )

    except FileNotFoundError as e:
        print(f"[-] CRITICAL: Input file not found - {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[-] CRITICAL: Invalid JSON syntax in input files - {e}")
        sys.exit(1)
    except ValidationError as e:
        print("[-] CRITICAL: Data structure validation failed!")
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"[-] CRITICAL: Unexpected error during parsing - {e}")
        sys.exit(1)
