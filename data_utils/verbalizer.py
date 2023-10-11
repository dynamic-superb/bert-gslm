import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict

from data_utils.global_constants import GlobalConstants


class Verbalizer:
    def __init__(self, method: str, data_dir: Path):
        random.seed(GlobalConstants.SEED)
        if method not in ["random", "freq"]:
            raise ValueError
        self.method = method
        self.data_dir = data_dir
        self.token_map = self.mapping()
        with (data_dir / "verbalizer.json").open(mode="w") as f:
            json.dump(self.token_map, f)

    def mapping(self) -> Dict:
        train_file = self.data_dir / "train.txt"
        if not train_file.exists():
            raise FileNotFoundError
        lines = train_file.open(mode="r").readlines()
        sep_token = GlobalConstants.SEP_TOKEN
        src, tgt = map(
            list, zip(*(line.split(sep_token) for line in lines if sep_token in line))
        )
        src_tokens = [token for s in src for token in s.split()]
        tgt_tokens = [token for s in tgt for token in s.split()]
        src_count = Counter(src_tokens)
        tgt_count = Counter(tgt_tokens)

        src_tokens = sorted(src_count, key=src_count.get, reverse=True)
        tgt_tokens = sorted(tgt_count, key=tgt_count.get, reverse=True)

        if self.method == "freq":
            assert len(src_tokens) >= len(tgt_tokens)

        if self.method == "random":
            src_tokens = random.sample(src_tokens, k=len(tgt_tokens))
            token_map = dict(zip(tgt_tokens, src_tokens))
        if self.method == "freq":
            assert len(src_tokens) >= len(tgt_tokens)
            token_map = dict(zip(tgt_tokens, src_tokens[: len(tgt_tokens)]))
        return token_map

    def verbalize(self, file_path: Path) -> None:
        lines = file_path.open(mode="r").readlines()
        sep_token = GlobalConstants.SEP_TOKEN
        src_lines, tgt_lines = map(
            list, zip(*(line.split(sep_token) for line in lines if sep_token in line))
        )
        with file_path.open(mode="w") as f:
            for src, tgt in zip(src_lines, tgt_lines):
                src = src.strip()
                tgt = tgt.strip()
                tgt_tokens = tgt.split()
                tgt_tokens = [self.token_map[tok] for tok in tgt_tokens]
                line = src + " " + sep_token + " " + (" ".join(tgt_tokens))
                f.write(f"{line}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
