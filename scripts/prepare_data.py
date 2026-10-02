"""Build the training sets of the nine paper runs from public sources.

VVR tasks come from the `vvr_easy` and `vvr_matched` configurations of the Hugging Face dataset
`stellalisy/VVRBench`. GenEval, OCR, and PickScore prompts come from the Flow-GRPO repository, and GenEval2
prompts from the official GenEval2 repository. UnifiedReward trains on the PickScore prompts. Each mixture
is rebuilt row by row, in the paper's order, from the lists in `data/idlists/`.

Usage:
    python scripts/prepare_data.py --out data            # all nine runs
    python scripts/prepare_data.py --out data --runs vvr_easy ocr_vvr_easy
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import urllib.request

HF_REPO = "stellalisy/VVRBench"
HF_REVISION = "f8609207c8ee846a2be0257f1c72a72ff276b9a4"
FLOW_GRPO = "https://raw.githubusercontent.com/yifan123/flow_grpo/879042cf5707f8b90daa98d147d7deac2317c5da/dataset"
GENEVAL2 = "https://raw.githubusercontent.com/facebookresearch/GenEval2/a6e82d2289e8d418f27f0adee77908b07060eea3"

# local name -> (URL, SHA256)
SOURCES = {
    "flow_grpo/geneval/train_metadata.jsonl": (
        f"{FLOW_GRPO}/geneval/train_metadata.jsonl",
        "34ff787e05efab6fb0c54742f5357d7a850360835818e401a4e55de77f8c9923",
    ),
    "flow_grpo/ocr/train.txt": (
        f"{FLOW_GRPO}/ocr/train.txt",
        "2eb32ce1c5fd1ffec94070be84252419ec848329d9adde9c2708867882fbce5f",
    ),
    "flow_grpo/pickscore/train.txt": (
        f"{FLOW_GRPO}/pickscore/train.txt",
        "2f425adcbb9a6d0feee7c6c448e1cb83a1f71deca9434b803907e2578eca837f",
    ),
    "geneval2/geneval2_data.jsonl": (
        f"{GENEVAL2}/geneval2_data.jsonl",
        "09233adbc9f877fba205f1a20efa4a8640e7e62a91dfd55893562bed8a928e43",
    ),
}

RUNS = [
    "vvr_easy",
    "vvr_matched",
    "geneval2",
    "ocr",
    "five_reward",
    "geneval2_vvr_easy",
    "geneval2_vvr_matched",
    "ocr_vvr_easy",
    "five_reward_vvr_easy",
]
IDLISTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "idlists")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_sources(cache):
    paths = {}
    for name, (url, sha) in SOURCES.items():
        path = os.path.join(cache, name)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path + ".tmp")
            os.replace(path + ".tmp", path)
        if _sha256(path) != sha:
            raise RuntimeError(f"{path} does not match the paper's source file (SHA256 {sha})")
        paths[name] = path
    return paths


def load_vvr(name):
    """Return {id: row} for one VVR training configuration."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        HF_REPO,
        f"training/{name}/train-00000-of-00001.parquet",
        repo_type="dataset",
        revision=HF_REVISION,
    )
    rows = {}
    for row in pq.read_table(path, columns=["id", "prompt", "complexity", "visual_spec_json"]).to_pylist():
        rows[row["id"]] = {
            "id": row["id"],
            "prompt": row["prompt"],
            "complexity": row["complexity"],
            "visual_spec": json.loads(row["visual_spec_json"]),
        }
    return rows


def load_sources(paths):
    def lines(name):
        with open(paths[name], encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    def jsonl(name):
        with open(paths[name], encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    return {
        "flow_grpo/geneval/train_metadata.jsonl": jsonl("flow_grpo/geneval/train_metadata.jsonl"),
        "flow_grpo/ocr/train.txt": [{"prompt": p.strip()} for p in lines("flow_grpo/ocr/train.txt")],
        "flow_grpo/pickscore/train.txt": [{"prompt": p.strip()} for p in lines("flow_grpo/pickscore/train.txt")],
        "geneval2/geneval2_data.jsonl": jsonl("geneval2/geneval2_data.jsonl"),
    }


def read_idlist(run):
    with gzip.open(os.path.join(IDLISTS, f"{run}.jsonl.gz"), "rt") as f:
        return [json.loads(line) for line in f]


def build_rows(run, sources, vvr):
    rows = []
    for entry in read_idlist(run):
        if entry["file"] == "vvr":
            row = dict(vvr[entry["training_source"]][entry["ref"]])
        else:
            row = dict(sources[entry["file"]][entry["ref"]])
        row["training_source"] = entry["training_source"]
        rows.append(row)
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data")
    parser.add_argument("--runs", nargs="+", default=RUNS, choices=RUNS)
    args = parser.parse_args()

    paths = fetch_sources(os.path.join(args.out, "sources"))
    sources = load_sources(paths)
    vvr = {}
    for run in args.runs:
        out_dir = os.path.join(args.out, run)
        if run in ("vvr_easy", "vvr_matched"):
            vvr.setdefault(run, load_vvr(run))
            rows = [{**row, "training_source": run} for row in vvr[run].values()]
        elif run == "ocr":
            # The OCR baseline reads Flow-GRPO's prompt file as is.
            os.makedirs(out_dir, exist_ok=True)
            shutil.copyfile(paths["flow_grpo/ocr/train.txt"], os.path.join(out_dir, "train.txt"))
            print(f"{run}: {len(sources['flow_grpo/ocr/train.txt'])} prompts")
            continue
        else:
            for name in ("vvr_easy", "vvr_matched"):
                if name in run:
                    vvr.setdefault(name, load_vvr(name))
            rows = build_rows(run, sources, vvr)
        write_jsonl(os.path.join(out_dir, "train_metadata.jsonl"), rows)
        counts = {}
        for row in rows:
            counts[row["training_source"]] = counts.get(row["training_source"], 0) + 1
        print(f"{run}: {len(rows)} prompts {counts}")


if __name__ == "__main__":
    main()
