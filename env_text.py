from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import signal
from tqdm import tqdm


def filter_emg_data(raw_data: np.ndarray, fs: int = 1000) -> np.ndarray:
    """
    Applies notch filters (50/150/250/350 Hz) + bandpass (10-400 Hz).
    Supports 1D or 2D arrays.
    """
    b1, a1 = signal.iirnotch(50, 30, fs)
    b2, a2 = signal.iirnotch(150, 30, fs)
    b3, a3 = signal.iirnotch(250, 30, fs)
    b4, a4 = signal.iirnotch(350, 30, fs)
    b5, a5 = signal.butter(4, [10 / (fs / 2), 400 / (fs / 2)], btype="bandpass")

    x = raw_data
    if x.ndim > 1:
        x = signal.filtfilt(b1, a1, x, axis=0)
        x = signal.filtfilt(b2, a2, x, axis=0)
        x = signal.filtfilt(b3, a3, x, axis=0)
        x = signal.filtfilt(b4, a4, x, axis=0)
        x = signal.filtfilt(b5, a5, x, axis=0)
    else:
        x = signal.filtfilt(b1, a1, x)
        x = signal.filtfilt(b2, a2, x)
        x = signal.filtfilt(b3, a3, x)
        x = signal.filtfilt(b4, a4, x)
        x = signal.filtfilt(b5, a5, x)

    return x


def build_text_train_json(
    base_dir: Path,
    output_json: Path,
    corpus_json: Path | None = None,
    prompt: str = "This EMG sequence corresponds to silent articulation. Predict the intended utterance and output the transcript only.",
    downsample_rate: int = 50,
    precision: int = 1,
    max_len_warning_threshold: int = 6000,  # ~ token count heuristic
) -> None:
    """
    Scans .mat files, converts EMG to compact text, and writes JSON items.
    Uses corpus.json to map label -> sentence (same logic as your image/video JSON builder).
    """
    print(f"[json] base_dir={base_dir}")

    # --- corpus.json load (same style as your first script) ---
    if corpus_json is None:
        corpus_json = base_dir / "corpus.json"
    if not corpus_json.exists():
        raise FileNotFoundError(f"corpus.json not found: {corpus_json}")
    print(f"[json] corpus_json={corpus_json}")

    with corpus_json.open("r", encoding="utf-8") as f:
        corpus = json.load(f)

    # --- scan mat files ---
    mat_files = sorted(base_dir.rglob("*.mat"))
    print(f"[json] mat_found={len(mat_files)}")

    items: list[dict] = []
    scanned = 0
    missing_sentence = 0
    failed = 0

    for mat_path in tqdm(mat_files, desc=f"Processing {base_dir.name}"):
        scanned += 1
        try:
            # --- ID generation (subject/session/label) ---
            rel = mat_path.relative_to(base_dir)
            label = mat_path.stem

            # expect: subject_x/sessionY/....../<label>.mat
            # your first script assumes rel.parts[0]=subject, rel.parts[1]=session
            subject = rel.parts[0] if len(rel.parts) >= 2 else "unknown_subject"
            session = rel.parts[1] if len(rel.parts) >= 2 else "unknown_session"
            item_id = f"{subject}_{session}_{label}"

            # --- sentence lookup (same as your first script) ---
            sentence = corpus.get(label)
            if sentence is None and label.isdigit():
                sentence = corpus.get(str(int(label)))
            if sentence is None:
                missing_sentence += 1
                continue

            # --- load & filter EMG ---
            mat_content = sio.loadmat(mat_path)
            if "data" not in mat_content:
                raise KeyError("missing key 'data' in .mat file")

            raw_emg = mat_content["data"].astype(np.float64)
            filtered = filter_emg_data(raw_emg)

            # --- downsample + stringify ---
            downsampled = filtered[::downsample_rate]
            flat = downsampled.flatten()
            emg_str_list = [f"{x:.{precision}f}" for x in flat]
            emg_text = ", ".join(emg_str_list)

            final_user_input = f"{prompt}\n\nEMG Sequence: [{emg_text}]"

            # --- truncation guard (same idea as your current script) ---
            # Rough: 1 token ~ 3-4 chars => threshold*3 chars heuristic
            char_cap = max_len_warning_threshold * 3
            if len(final_user_input) > char_cap:
                final_user_input = final_user_input[:char_cap] + " ...]"

            items.append(
                {
                    "id": item_id,
                    "conversations": [
                        {"from": "human", "value": final_user_input},
                        {"from": "gpt", "value": sentence},
                    ],
                }
            )

        except Exception as e:
            failed += 1
            print(f"[warn] could not process: {mat_path} ({e})")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(
        "[json] scanned_mat={} missing_sentence={} failed={} saved={}".format(
            scanned, missing_sentence, failed, len(items)
        )
    )


def create_text_train_json(
    base_dir: Path = Path("Resource/data"),
    output_json: Path | None = None,
    output_dir: Path | None = None,
    corpus_json: Path | None = None,
    prompt: str = "This EMG sequence corresponds to silent articulation. Predict the intended utterance and output the transcript only.",
    downsample_rate: int = 50,
    precision: int = 1,
) -> Path | list[Path]:
    """
    Mirrors your create_train_json logic:
    - If base_dir has Train/Val/Test folders, create train.json/val.json/test.json in output_dir
    - Otherwise, create a single train.json under base_dir (or output_json)
    """
    split_names = ("Train", "Val", "Test")
    split_dirs = {name: base_dir / name for name in split_names}

    # Split-mode
    if any(p.is_dir() for p in split_dirs.values()):
        if output_dir is None:
            output_dir = base_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # IMPORTANT: corpus.json is shared at base_dir level (same as your first script)
        corpus_path = corpus_json or (base_dir / "corpus.json")
        if not corpus_path.exists():
            raise FileNotFoundError(f"corpus.json not found: {corpus_path}")

        outputs: list[Path] = []
        for name, split_dir in split_dirs.items():
            if not split_dir.is_dir():
                print(f"[json] skip {name}: not found -> {split_dir}")
                continue

            out_path = output_dir / f"{name.lower()}.json"
            build_text_train_json(
                base_dir=split_dir,
                output_json=out_path,
                corpus_json=corpus_path,
                prompt=prompt,
                downsample_rate=downsample_rate,
                precision=precision,
            )
            outputs.append(out_path)

        return outputs

    # Single-dir mode
    if output_json is None:
        output_json = base_dir / "train.json"

    build_text_train_json(
        base_dir=base_dir,
        output_json=output_json,
        corpus_json=corpus_json,
        prompt=prompt,
        downsample_rate=downsample_rate,
        precision=precision,
    )
    return output_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert EMG .mat to JSON (corpus.json -> transcript).")
    parser.add_argument("--base-dir", type=Path, default=Path("Resource/data"), help="Base data directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (defaults to base-dir)")
    parser.add_argument("--output-json", type=Path, default=None, help="Output JSON path (single-dir mode)")
    parser.add_argument("--corpus-json", type=Path, default=None, help="corpus.json path (defaults to base-dir/corpus.json)")
    parser.add_argument(
        "--prompt",
        type=str,
        default="This EMG sequence corresponds to silent articulation. Predict the intended utterance and output the transcript only.",
        help="Instruction prompt",
    )
    parser.add_argument("--downsample", type=int, default=50, help="Downsample rate (keep 1 out of N)")
    parser.add_argument("--precision", type=int, default=1, help="Floating point precision")

    args = parser.parse_args()

    create_text_train_json(
        base_dir=args.base_dir,
        output_json=args.output_json,
        output_dir=args.output_dir,
        corpus_json=args.corpus_json,
        prompt=args.prompt,
        downsample_rate=args.downsample,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
