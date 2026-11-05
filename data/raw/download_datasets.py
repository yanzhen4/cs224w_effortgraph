"""Download pinned official benchmark repositories and inventory their records.

Requires curl and pyarrow. Run from any directory with python3.
Raw repository files are preserved; JSONL exports retain all original fields.
"""
import collections
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
REPOS = {
    "mmlu_pro": "TIGER-Lab/MMLU-Pro",
    "supergpqa": "m-a-p/SuperGPQA",
    **{f"livebench/{c}": f"livebench/{c}" for c in (
        "coding", "data_analysis", "instruction_following", "math", "reasoning", "language"
    )},
}


def fetch(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    subprocess.run(["curl", "-fLsS", "--retry", "4", "--connect-timeout", "30",
                    "--max-time", "600", url, "-o", str(temporary)], check=True)
    temporary.replace(destination)


def serialize(value):
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def download_repository(item):
    name, repo = item
    directory = ROOT / "data/raw" / name
    metadata_path = directory / "_hf_metadata.json"
    # Reuse the initial metadata to keep reruns pinned to the same snapshot.
    if not metadata_path.exists():
        fetch(f"https://huggingface.co/api/datasets/{repo}", metadata_path)
    metadata = json.loads(metadata_path.read_text())
    revision = metadata["sha"]
    result = {"repository": repo, "revision": revision, "files": [], "records": 0}
    for entry in metadata["siblings"]:
        relative = entry["rfilename"]
        path = directory / relative
        url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(relative)}"
        if not path.exists():
            fetch(url, path)
        info = {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "url": url}
        rows = None
        if path.suffix == ".parquet":
            rows = pq.read_table(path).to_pylist()
        elif path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if rows is not None:
            info["records"] = len(rows)
            result["records"] += len(rows)
            info["fields"] = sorted({key for row in rows for key in row})
            info["distributions"] = {}
            for field in ("category", "task", "discipline", "field", "subfield", "difficulty",
                          "livebench_release_date", "livebench_removal_date"):
                if any(field in row for row in rows):
                    info["distributions"][field] = dict(sorted(collections.Counter(
                        str(row.get(field)) for row in rows).items()))
            export = directory / "jsonl" / (path.stem + ".jsonl")
            export.parent.mkdir(parents=True, exist_ok=True)
            with export.open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, default=serialize) + "\n")
            info["jsonl_export"] = str(export.relative_to(ROOT))
        result["files"].append(info)
    print(f"{name}: {result['records']} records, revision {revision}", flush=True)
    return name, result


def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        repositories = dict(pool.map(download_repository, REPOS.items()))
    totals = {
        "mmlu_pro": repositories["mmlu_pro"]["records"],
        "supergpqa": repositories["supergpqa"]["records"],
        "livebench": sum(r["records"] for n, r in repositories.items() if n.startswith("livebench/")),
    }
    manifest = {"completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "scope": "All files and splits of the eight official question repositories; no filtering or deduplication.",
                "totals": totals, "repositories": repositories}
    (ROOT / "data/raw/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
