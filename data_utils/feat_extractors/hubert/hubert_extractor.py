import argparse
import joblib
import yaml
from itertools import groupby
from typing import List, Union

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from torch import Tensor

from data_utils.downloader import Downloader
from data_utils.feat_extractors import ExtractorBase
from data_utils.global_constants import GlobalConstants


class Extractor(ExtractorBase):
    def __init__(self, layer: int = 6, max_chunk: int = 1600000) -> None:
        self.download()

        model, _, task = load_model_ensemble_and_task(
            [str(GlobalConstants.HUBERT_MODEL_PATH)]
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model[0].eval().to(self.device)
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk

        self.kmeans = joblib.load(GlobalConstants.HUBERT_KMEANS_PATH.open(mode="rb"))
        self.kmeans.verbose = False

        dict_path = GlobalConstants.HUBERT_LM_PATH.parent / "dict.txt"
        if not dict_path.exists():
            raise FileNotFoundError
        lines = dict_path.open(mode="r").readlines()
        self.token_size = len(lines)

    def extract_feat(
        self, wav: np.ndarray, sr: int, kmeans: bool = True, reduce: bool = False
    ) -> Union[Tensor, List[int]]:
        if wav.ndim == 2:
            wav = wav.mean(axis=-1)
        if sr != self.task.cfg.sample_rate:
            wav = librosa.resample(wav, sr, self.task.cfg.sample_rate)
        with torch.no_grad():
            wav = wav.astype(np.float32)
            wav = torch.from_numpy(wav).to(self.device)
            if self.task.cfg.normalize:
                wav = F.layer_norm(wav, wav.shape)
            wav = wav.view(1, -1)
            feat_list = []
            for start in range(0, wav.size(1), self.max_chunk):
                chunk = wav[:, start : (start + self.max_chunk)]
                feat, _ = self.model.extract_features(
                    source=chunk, padding_mask=None, mask=False, output_layer=self.layer
                )
            feat_list.append(feat)
        feats = torch.cat(feat_list, dim=1).squeeze(0)
        if kmeans:
            feats = self.kmeans.predict(feats.cpu().numpy())
            feats = feats.tolist()
            if reduce:
                feats = [item[0] for item in groupby(feats)]
        return feats

    def download(self):
        hubert_url = GlobalConstants.HUBERT_MODEL_URL
        hubert_path = GlobalConstants.HUBERT_MODEL_PATH
        kmeans_url = GlobalConstants.HUBERT_KMEANS_URL
        kmeans_path = GlobalConstants.HUBERT_KMEANS_PATH
        gslm_url = GlobalConstants.HUBERT_LM_URL
        gslm_path = GlobalConstants.HUBERT_LM_PATH
        Downloader.download(hubert_url, hubert_path)
        Downloader.download(kmeans_url, kmeans_path)
        Downloader.download(gslm_url, gslm_path)
        Downloader.extract_tgz(gslm_path, gslm_path.parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: Complete UT.
    extractor = Extractor()
    extractor.download()
