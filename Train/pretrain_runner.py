#!/usr/bin/env python3
import sys
import argparse
import subprocess
from pathlib import Path

from pretrain_download import HuggingFaceDownload


def quote_cmd(cmd: list[str]) -> str:
    # Quote args that contain spaces/tabs for readable printing
    return " ".join([f'"{c}"' if (" " in c or "\t" in c) else c for c in cmd])


def is_subpath(child: Path, parent: Path) -> bool:
    # Return True if 'child' is inside 'parent' (or equal), without requiring Python 3.9+ is_relative_to
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def paths_overlap(a: Path, b: Path) -> bool:
    # Return True if two directories overlap (one contains the other) or are identical
    a = a.resolve()
    b = b.resolve()
    return a == b or is_subpath(a, b) or is_subpath(b, a)


def build_cmd(args, run_py: Path) -> list[str]:
    # Build the exact CLI command that will be passed to run.py
    cmd = [
        sys.executable, str(run_py),
        "--model-id", str(args.model_path),
        "--data-path", str(args.data_path),
        "--image-folder", str(args.image_folder),
        "--output-dir", str(args.output_dir),
        "--per-device-train-batch-size", str(args.per_device_train_batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),

        # macOS-friendly flags
        "--disable-flash-attn2",

        # LoRA stage-2 flags
        "--lora-enable",
        "--lora-rank", str(args.lora_rank),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),

        # Freeze the projector/merger for stage-2 LoRA (recommended)
        "--freeze-merger",
    ]

    # Optional parameters
    if args.dataloader_num_workers is not None:
        cmd += ["--dataloader-num-workers", str(args.dataloader_num_workers)]
    if args.num_train_epochs is not None:
        cmd += ["--num-train-epochs", str(args.num_train_epochs)]
    if args.learning_rate is not None:
        cmd += ["--learning-rate", str(args.learning_rate)]

    return cmd


def main():
    p = argparse.ArgumentParser()

    # If --download-hf is provided, model-path is optional and will be overwritten by hf-download-dir
    p.add_argument(
        "--model-path",
        default=None,
        help="Local projector-trained model dir or checkpoint-xxxx dir. Optional if --download-hf is set.",
    )

    # Dataset paths
    p.add_argument("--data-path", default="Resource/data/train.json")
    p.add_argument("--image-folder", default="Resource/data")

    # Output path
    p.add_argument("--output-dir", default="output/lora_on_projector_2b")

    # Training hyperparameters
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=2)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Misc
    p.add_argument("--dataloader-num-workers", type=int, default=8)
    p.add_argument("--num-train-epochs", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--dry-run", action="store_true", help="Print command only; do not execute")

    # Optional: download HF repo and use it as --model-id input automatically
    p.add_argument("--download-hf", action="store_true", help="Download HF repo (diddmstjr/ISEF) and use as model input")
    p.add_argument("--hf-download-dir", default="pretrained/ISEF", help="Directory to store HF download")

    args = p.parse_args()

    # Resolve paths that may be used later
    args.data_path = Path(args.data_path).expanduser().resolve()
    args.image_folder = Path(args.image_folder).expanduser().resolve()
    args.output_dir = Path(args.output_dir).expanduser().resolve()

    if not args.data_path.exists():
        raise FileNotFoundError(f"data_path not found: {args.data_path}")
    if not args.image_folder.exists():
        raise FileNotFoundError(f"image_folder not found: {args.image_folder}")

    # run.py is located in the parent directory of this launcher script's directory
    run_py = Path(__file__).resolve().parent.parent / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"run.py not found at: {run_py}")

    # If requested, download the repo and use its directory as model input
    if args.download_hf:
        hf_dir = Path(args.hf_download_dir).expanduser().resolve()
        hf_dir.mkdir(parents=True, exist_ok=True)

        # If user also provided a model-path, prevent overlap to avoid accidental mixing
        if args.model_path:
            tmp_model_path = Path(args.model_path).expanduser().resolve()
            if paths_overlap(hf_dir, tmp_model_path):
                raise ValueError(
                    f"hf_download_dir ({hf_dir}) overlaps with model_path ({tmp_model_path}). "
                    "Choose separate directories to avoid overwrite/mixing."
                )

        downloader = HuggingFaceDownload()
        downloader.download_full_repo(local_dir=str(hf_dir))
        print(f"[HF] Downloaded repo into: {hf_dir}")

        # Use the downloaded directory as the model input path
        args.model_path = hf_dir

    # If not downloading from HF, model-path must exist locally
    if not args.model_path:
        args.model_path = input(
            "Enter local projector model path (e.g., output/projector_ft_2b or .../checkpoint-xxx): "
        ).strip()

    args.model_path = Path(args.model_path).expanduser().resolve()
    if not args.model_path.exists():
        raise FileNotFoundError(f"model_path not found: {args.model_path}")

    # Build and print command
    cmd = build_cmd(args, run_py)

    print("\n[CMD]")
    print(quote_cmd(cmd))
    print()

    if args.dry_run:
        return

    # Run from run.py directory so any relative paths inside run.py behave consistently
    proc = subprocess.run(cmd, check=False, cwd=str(run_py.parent))
    if proc.returncode != 0:
        raise SystemExit(f"run.py failed with return code {proc.returncode}")


if __name__ == "__main__":
    main()
