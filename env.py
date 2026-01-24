from __future__ import annotations

import argparse
from pathlib import Path
from module.helper import create_train_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train.json for EMG spectrogram + video dataset."
    )
    parser.add_argument("--base-dir", type=Path, default=Path("Resource/data"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--corpus-json", type=Path, default=None)
    parser.add_argument(
        "--prompt",
        type=str,
        default="<image>\\n<video>\\nTranscribe the spoken sentence.",
    )
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--video-frame-count", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = create_train_json(
        base_dir=args.base_dir,
        output_json=args.output_json,
        output_dir=args.output_dir,
        corpus_json=args.corpus_json,
        prompt=args.prompt,
        preprocess=args.preprocess,
        video_frame_count=args.video_frame_count,
    )
    if isinstance(output_paths, list):
        for path in output_paths:
            print(f"Saved: {path}")
    else:
        print(f"Saved: {output_paths}")


if __name__ == "__main__":
    main()
