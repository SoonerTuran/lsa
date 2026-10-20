"""Idempotent Colab launcher: clone/pull the repo, pip install -e, mount Drive, train.

Run it *inside the notebook kernel* (so Google Drive can be mounted):

    %run /content/lsa/colab/run_colab.py --config configs/phase1.yaml

Re-running is safe: the repo is fast-forwarded, the install is a no-op, and training
resumes from <out>/checkpoints/last.pt when it exists (or only re-does the final
evaluation if training already finished). For a private repo, add a Colab secret named
GITHUB_TOKEN (or set the env var) and it is used for cloning.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO = os.environ.get("LSA_REPO", "https://github.com/SoonerTuran/lsa.git")
DEFAULT_REPO_DIR = "/content/lsa"
DRIVE_MOUNT = "/content/drive"


def run(cmd: list[str], cwd: str | None = None, dry: bool = False, secret: str | None = None) -> None:
    """Run a command and stream its output into the notebook (fd-level output of child
    processes is not shown by Colab otherwise)."""
    shown = " ".join(cmd)
    if secret:
        shown = shown.replace(secret, "***")
    print(f"$ {shown}", flush=True)
    if dry:
        return
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line if not secret else line.replace(secret, "***"), end="", flush=True)
    if proc.wait() != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {shown}")


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def github_token() -> str | None:
    token = os.environ.get("GITHUB_TOKEN")
    if token or not in_colab():
        return token
    try:
        from google.colab import userdata

        return userdata.get("GITHUB_TOKEN")
    except Exception:
        return None


def mount_drive() -> None:
    if os.path.isdir(os.path.join(DRIVE_MOUNT, "MyDrive")):
        print("Drive already mounted")
        return
    from google.colab import drive

    drive.mount(DRIVE_MOUNT)


def sync_repo(url: str, dest: str, branch: str, dry: bool) -> None:
    token = github_token()
    auth_url = url.replace("https://", f"https://x-access-token:{token}@", 1) if token and url.startswith("https://") else url
    if os.path.isdir(os.path.join(dest, ".git")):
        run(["git", "-C", dest, "pull", "--ff-only", auth_url, branch], dry=dry, secret=token)
    else:
        run(["git", "clone", "--branch", branch, auth_url, dest], dry=dry, secret=token)
        # keep the token out of .git/config
        run(["git", "-C", dest, "remote", "set-url", "origin", url], dry=dry)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/phase1.yaml", help="path relative to the repo")
    ap.add_argument("--out", default=None, help="output dir (default: Drive/lsa_runs/<config name>)")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--branch", default="main")
    ap.add_argument("--repo-dir", default=DEFAULT_REPO_DIR)
    ap.add_argument("--no-drive", action="store_true", help="do not mount Drive; default out under /content")
    ap.add_argument("--skip-sync", action="store_true", help="use --repo-dir as is (no clone/pull/install)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides passed to lsa.train")
    args = ap.parse_args(argv)

    use_drive = in_colab() and not args.no_drive
    if use_drive and not args.dry_run:
        mount_drive()
    name = Path(args.config).stem
    out = args.out or (f"{DRIVE_MOUNT}/MyDrive/lsa_runs/{name}" if use_drive else f"/content/lsa_runs/{name}")

    if not args.skip_sync:
        sync_repo(args.repo, args.repo_dir, args.branch, args.dry_run)
        run([sys.executable, "-m", "pip", "install", "-q", "-e", args.repo_dir], dry=args.dry_run)

    ckpt = Path(out) / "checkpoints" / "last.pt"
    print(f"output dir: {out}  ({'resuming from ' + str(ckpt) if ckpt.exists() else 'fresh run'})", flush=True)
    cmd = [sys.executable, "-m", "lsa.train", "--config", args.config, "--out", out]
    if args.set:
        cmd += ["--set", *args.set]
    run(cmd, cwd=args.repo_dir, dry=args.dry_run)

    metrics = Path(out) / "metrics.json"
    if metrics.exists():
        print(json.dumps(json.loads(metrics.read_text()), indent=2))


if __name__ == "__main__":
    main()
