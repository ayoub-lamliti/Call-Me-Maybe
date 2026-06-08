from pydantic import BaseModel, ValidationError
from typing import Any
import argparse
import json
import sys
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class FunctionParameter(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, FunctionParameter] = {}
    returns: dict[str, Any] = {}


class Prompt(BaseModel):
    prompt: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Function Calling Inference")
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
    )
    return parser.parse_args()


def parse_input_files(
    args: argparse.Namespace, model: Small_LLM_Model
) -> tuple[dict, str, list]:
    try:
        with open(args.functions_definition, "r") as f:
            raw_funcs = json.load(f)
        validated_funcs = [FunctionDefinition(**obj) for obj in raw_funcs]
        list_of_functions = {func.name: func.model_dump() for func in validated_funcs}
        list_of_decode_name_functions = []
        for function in list_of_functions:
            list_of_decode_name_functions.extend([model.encode(function)[0].tolist()])
        functions_tools = "\n".join(json.dumps(obj) for obj in raw_funcs)
        with open(args.input, "r") as f:
            raw_prompts = json.load(f)
        validated_prompts = [Prompt(**obj).model_dump() for obj in raw_prompts]
        return (
            list_of_functions,
            functions_tools,
            validated_prompts,
            list_of_decode_name_functions,
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
