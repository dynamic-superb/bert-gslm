from pathlib import Path


class GlobalConstants:
    # URLs for downloading models.
    HUBERT_MODEL_URL = "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"

    HUBERT_KMEANS_URL = (
        "https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/km100/km.bin"
    )

    HUBERT_LM_URL = "https://dl.fbaipublicfiles.com/textless_nlp/gslm/hubert/lm_km100/hubert100_lm.tgz"

    # Save paths for models.
    SAVE_ROOT = Path(".")

    HUBERT_MODEL_PATH = SAVE_ROOT / "HuBERT" / "hubert_base_ls960.pt"

    HUBERT_KMEANS_PATH = SAVE_ROOT / "HuBERT" / "kmeans.bin"

    HUBERT_LM_PATH = SAVE_ROOT / "HuBERT" / "hubert100_lm.tgz"

    # Token sizes.
    HUBERT_LM_ORG_TOKEN_SIZE = 100

    SEP_TOKEN = "<s>"

    SEED = 42

    def __init__(self):
        raise NotImplementedError
