import yaml
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer

from data_utils.global_constants import GlobalConstants
from data_utils.preprocessors import PreprocessorBase
from data_utils.verbalizer import Verbalizer


class Preprocessor(PreprocessorBase):
    def __init__(
        self,
        json_path: Path,
        split: str,
        ext_type: str,
        tok_type: str,
        vb_method: Optional[str] = None,
    ) -> None:
        # TODO: Add config file for extractor.
        super().__init__(json_path, split)
        self.feat_extractor = self.load_extractor(ext_type)
        ext_config_path = Path(__file__).resolve().parent / "config.yaml"
        if not ext_config_path.exists():
            raise FileNotFoundError
        self.ext_config = yaml.load(
            ext_config_path.open(mode="r"), Loader=yaml.FullLoader
        )[ext_type]
        self.tokenizer = AutoTokenizer.from_pretrained(tok_type)
        self.is_natural_label = False
        self.vb_method = vb_method

    def process(self, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        self.dump_token_dict(save_dir / "dict.txt")
        for dataset in self.dataset_list:
            feat_list = []
            name_list = []
            for sample in self.pbar(dataset):
                audio = sample["audio"]["array"]
                sr = sample["audio"]["sampling_rate"]
                label = sample["label"]
                instr = sample["instruction"]
                name = sample["file"]
                speech_ids = self.feat_extractor.extract_feat(
                    audio, sr, **self.ext_config
                )
                instr_tokens = self.tokenizer.tokenize(instr)
                instr_ids = self.tokenizer.convert_tokens_to_ids(instr_tokens)
                instr_ids = [
                    tok_id + self.feat_extractor.token_size for tok_id in instr_ids
                ]
                feat_ids = instr_ids + speech_ids
                feat_ids = [str(tok) for tok in feat_ids]
                if self.is_natural_label:
                    label_tokens = self.tokenizer.tokenize(label)
                    label_ids = self.tokenizer.convert_tokens_to_ids(label_tokens)
                    label_ids = [
                        tok_id + self.feat_extractor.token_size for tok_id in label_ids
                    ]
                    label_ids = [str(tok) for tok in label_ids]
                    feat = feat_ids + [GlobalConstants.SEP_TOKEN] + label_ids
                else:
                    feat = feat_ids + [GlobalConstants.SEP_TOKEN] + [str(label)]
                feat_list.append(feat)
                name_list.append(name)
            save_path = save_dir / f"{dataset.split}.txt"
            with save_path.open(mode="w") as f:
                for feat in feat_list:
                    line = " ".join(feat)
                    f.write(f"{line}\n")
            if dataset.split == "test":
                name_path = save_dir / "test_files.txt"
                with name_path.open(mode="w") as f:
                    for name in name_list:
                        f.write(f"{name}\n")

        if not self.is_natural_label:
            verbalizer = Verbalizer(self.vb_method, save_dir)
            for dataset in self.dataset_list:
                verbalizer.verbalize(save_dir / f"{dataset.split}.txt")

    def dump_token_dict(self, save_path: Path) -> None:
        with save_path.open(mode="w") as f:
            token_size = self.feat_extractor.token_size + self.tokenizer.vocab_size
            for index in range(token_size):
                f.write(f"{index + 1} 1\n")
