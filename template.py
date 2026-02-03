import os
from pathlib import Path

PROJECT_NAME = "mini-gpt"

PROJECT_STRUCTURE = [
    "requirements.txt",
    "setup.py",

    "configs/model_config.yaml",

    "data/raw/.gitkeep",
    "data/processed/.gitkeep",

    "notebooks/mini_gpt_experiments.ipynb",

    "src/mini_gpt/__init__.py",

    "src/mini_gpt/models/__init__.py",
    "src/mini_gpt/models/gpt.py",
    "src/mini_gpt/models/transformer_block.py",
    "src/mini_gpt/models/attention.py",
    "src/mini_gpt/models/feed_forward.py",
    "src/mini_gpt/models/layers.py",

    "src/mini_gpt/data/__init__.py",
    "src/mini_gpt/data/dataset.py",
    "src/mini_gpt/data/dataloader.py",

    "src/mini_gpt/training/__init__.py",
    "src/mini_gpt/training/trainer.py",
    "src/mini_gpt/training/evaluator.py",

    "src/mini_gpt/inference/__init__.py",
    "src/mini_gpt/inference/generate.py",

    "src/mini_gpt/utils/__init__.py",
    "src/mini_gpt/utils/tokenizer.py",
    "src/mini_gpt/utils/weights.py",

    "src/mini_gpt/main.py",

    "scripts/train.py",
    "scripts/generate_text.py",

    "tests/test_model.py",
]

def create_project_structure():
    print(f"Creating project structure for: {PROJECT_NAME}\n")

    for path in PROJECT_STRUCTURE:
        full_path = Path(path)

      
        if path.endswith("/"):
            full_path.mkdir(parents=True, exist_ok=True)
            continue

        full_path.parent.mkdir(parents=True, exist_ok=True)

        if not full_path.exists():
            full_path.touch()
            print(f" Created: {full_path}")

    print("\n Project structure created successfully!")

if __name__ == "__main__":
    create_project_structure()
