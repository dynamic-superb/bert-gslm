import requests
import tarfile
from pathlib import Path
from typing import Generator

from tqdm import tqdm


class Downloader:
    def __init__(self):
        raise NotImplementedError

    @classmethod
    def download(cls, url: str, save_path: Path, force: bool = False) -> None:
        if not save_path.exists() or force:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with requests.get(url, stream=True) as rq:
                total_length = int(rq.headers.get("content-length"))
                with save_path.open(mode="wb") as f:
                    for chunk in cls._get_chunks(rq, total_length):
                        f.write(chunk)

    @classmethod
    def extract_tgz(cls, tgz_path: Path, dst_dir: Path) -> None:
        tgz = tarfile.open(tgz_path)
        for member in tgz.getmembers():
            if member.isreg():
                member.name = Path(member.name).name
                tgz.extract(member, dst_dir)

    @classmethod
    def _get_chunks(
        cls, response: requests.Response, total_length: int, chunk_size: int = 1024
    ) -> Generator[bytes, None, None]:
        pbar = tqdm(total=total_length, unit="B", unit_scale=True)
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                yield chunk
                pbar.update(len(chunk))
        pbar.close()
