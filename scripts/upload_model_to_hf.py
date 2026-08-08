"""Upload the trained model to a Hugging Face Hub model repo.

Run this yourself after logging in (`huggingface-cli login`):

    python scripts/upload_model_to_hf.py --repo-id <your-username>/leakdb-hanoi-rf

By default this uploads rf_model_compressed.joblib (the smaller of the two
saved models — same predictions, less bandwidth). The repo is created if it
doesn't exist yet.
"""

import argparse
import os

from huggingface_hub import HfApi

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id", required=True, help="e.g. your-username/leakdb-hanoi-rf"
    )
    parser.add_argument(
        "--file",
        default="rf_model_compressed.joblib",
        help="Model filename in the repo root to upload",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    local_path = os.path.join(REPO_ROOT, args.file)
    if not os.path.exists(local_path):
        raise SystemExit(f"File not found: {local_path}")

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=args.file,
        repo_id=args.repo_id,
    )
    print(f"Uploaded {args.file} to https://huggingface.co/{args.repo_id}")
    print(f"Now set HF_REPO_ID={args.repo_id} when running the app.")


if __name__ == "__main__":
    main()
