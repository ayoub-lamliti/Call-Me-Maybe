install:
	@uv sync

run:
	@uv run python -m src
	echo "running..."

debug:
	@uv run python -m pdb src/__main__.py

clean:
	@rm -rf __pycache__
	@rm -rf src/__pycache__
	@rm -rf .mypy_cache
	@rm -rf .pytest_cache
	@find . -type d -name "__pycache__" -exec rm -r {} +
	echo "all clean."

lint:
	flake8 src/
	mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint
