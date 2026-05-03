"""
Upload Files to Private HuggingFace Repository
================================================
Uploads all output files from the Arabic->Turkish pipeline
to a private HuggingFace dataset repository.

Install:
    pip install huggingface_hub

Usage:
    python upload_to_hf.py --folder "C:\\path\\to\\output\\folder" --repo "your-username/your-repo-name"

Arguments:
    --folder   Path to the folder containing the files to upload
    --repo     HuggingFace repo in format username/repo-name (will be created if it doesn't exist)
    --token    Your HuggingFace token (or set HF_TOKEN environment variable)
    --private  Make the repository private (default: True)

Getting your HuggingFace token:
    1. Go to https://huggingface.co and sign in
    2. Click your profile picture → Settings → Access Tokens
    3. Click "New token" → name it anything → role: "write" → Generate
    4. Copy the token (starts with hf_...)
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Upload files to HuggingFace")
    parser.add_argument("--folder",  required=True,
                        help="Path to folder containing files to upload")
    parser.add_argument("--repo",    required=True,
                        help="HuggingFace repo name, e.g. hanaa/arabic-turkish-video")
    parser.add_argument("--token",   default=None,
                        help="HuggingFace write token (or set HF_TOKEN env variable)")
    parser.add_argument("--private", action="store_true", default=True,
                        help="Make repo private (default: True)")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("Run: pip install huggingface_hub")
        sys.exit(1)

    # ── Resolve token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("\nNo token provided.")
        print("Either pass --token hf_xxxx  or set the HF_TOKEN environment variable.")
        print("\nTo get a token:")
        print("  1. Go to https://huggingface.co → Settings → Access Tokens")
        print("  2. Create a new token with 'write' role")
        sys.exit(1)

    # ── Validate folder
    if not os.path.isdir(args.folder):
        print(f"Error: Folder not found: {args.folder}")
        sys.exit(1)

    # ── Collect files to upload (skip hidden files and .json transcript if desired)
    files = [
        f for f in os.listdir(args.folder)
        if os.path.isfile(os.path.join(args.folder, f))
        and not f.startswith(".")
    ]

    if not files:
        print(f"No files found in: {args.folder}")
        sys.exit(1)

    print(f"\nFiles to upload ({len(files)}):")
    for f in files:
        size_mb = os.path.getsize(os.path.join(args.folder, f)) / 1_048_576
        print(f"  {f}  ({size_mb:.1f} MB)")

    # ── Create repo if it doesn't exist
    api = HfApi(token=token)

    print(f"\nCreating/checking repository: {args.repo}")
    try:
        create_repo(
            repo_id=args.repo,
            repo_type="dataset",   # dataset repos have no file size limits
            private=args.private,
            token=token,
            exist_ok=True          # don't error if already exists
        )
        visibility = "private" if args.private else "public"
        print(f"  Repository ready ({visibility}): https://huggingface.co/datasets/{args.repo}")
    except Exception as e:
        print(f"Error creating repo: {e}")
        sys.exit(1)

    # ── Upload files one by one
    print(f"\nUploading files...")
    failed = []

    for filename in files:
        local_path = os.path.join(args.folder, filename)
        size_mb    = os.path.getsize(local_path) / 1_048_576
        print(f"\n  Uploading: {filename}  ({size_mb:.1f} MB)")

        try:
            url = api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=filename,
                repo_id=args.repo,
                repo_type="dataset",
                token=token,
            )
            print(f"  Done -> {url}")
        except Exception as e:
            print(f"  Failed: {e}")
            failed.append(filename)

    # ── Summary
    print("\n" + "=" * 60)
    succeeded = len(files) - len(failed)
    print(f"  Uploaded {succeeded}/{len(files)} files")
    print(f"  Repository: https://huggingface.co/datasets/{args.repo}")
    if failed:
        print(f"\n  Failed files:")
        for f in failed:
            print(f"    - {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
