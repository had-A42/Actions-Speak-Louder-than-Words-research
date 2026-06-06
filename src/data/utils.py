from pathlib import Path

import requests
from tqdm import tqdm

def download_file(
    url: str,
    output_path: str | Path,
    overwrite: bool = False,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        print(f"File already exists: {output_path}")
        return output_path

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(tmp_path, "wb") as file:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=output_path.name,
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))

    tmp_path.replace(output_path)

    print(f"Downloaded to: {output_path}")
    return output_path
