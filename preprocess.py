import argparse
import importlib
from pathlib import Path
from typing import Optional


def main(
    json_path: Path,
    save_dir: Path,
    task_name: str,
    split: str,
    vb_method: Optional[str] = None,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        task_module = importlib.import_module(f"data_utils.preprocessors.{task_name}")
        Preprocessor = task_module.Preprocessor
    except:
        raise ImportError(f"Failed to Preprocessor from task: {task_name}.")

    preprocessor = Preprocessor(
        json_path, split, "hubert", "bert-base-uncased", vb_method
    )
    preprocessor.process(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("save_dir", type=Path)
    parser.add_argument("task_name", type=str)
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--vb_method", "-vb", type=str, default=None)
    main(**vars(parser.parse_args()))
