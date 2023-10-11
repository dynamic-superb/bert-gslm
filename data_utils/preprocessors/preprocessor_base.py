import importlib
import json
from pathlib import Path
from typing import Type

from datasets import load_dataset
from tqdm import tqdm

from data_utils.feat_extractors import ExtractorBase


class PreprocessorBase:
    def __init__(
        self, json_path: Path, split: str = "train", progress_bar: bool = True
    ) -> None:
        config = json.load(json_path.open(mode="r"))
        if split not in ["train", "validation", "test", "all"]:
            raise ValueError(
                f"Expected split to be in ['train', 'validation', 'test', 'all'], but got {split}."
            )
        if split != "all":
            self.dataset_list = [
                load_dataset(config["path"], revision=config["version"], split=split)
            ]
        else:
            self.dataset_list = [
                load_dataset(config["path"], revision=config["version"], split=sp)
                for sp in ["train", "validation", "test"]
            ]

        self.pbar = tqdm if progress_bar else lambda x: x

    def load_extractor(self, ext_type: str) -> Type[ExtractorBase]:
        try:
            ext_module = importlib.import_module(
                f"data_utils.feat_extractors.{ext_type}"
            )
            Extractor = ext_module.Extractor
        except:
            raise ImportError(
                f"Failed to import data_utils.feat_extractors.{ext_type}."
            )
        extractor = Extractor()
        return extractor

    def process(self, save_dir: Path) -> None:
        raise NotImplementedError
