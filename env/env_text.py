import os
import json
import numpy as np
import scipy.io as sio
from scipy import signal
from pathlib import Path
import argparse
from tqdm import tqdm

def filter_emg_data(raw_data: np.ndarray) -> np.ndarray:
    """
    Applies a series of notch and bandpass filters to the raw EMG data,
    mimicking the preprocessing from the original repository.
    """
    fs = 1000
    # Create notch filters to remove powerline interference
    b1, a1 = signal.iirnotch(50, 30, fs)
    b2, a2 = signal.iirnotch(150, 30, fs)
    b3, a3 = signal.iirnotch(250, 30, fs)
    b4, a4 = signal.iirnotch(350, 30, fs)
    # Create a Butterworth band-pass filter
    b5, a5 = signal.butter(4, [10 / (fs / 2), 400 / (fs / 2)], 'bandpass')

    # Apply filters sequentially
    # The original data seems to be processed per channel, assuming shape [samples, channels]
    # We will filter along axis 0.
    if raw_data.ndim > 1:
        x = signal.filtfilt(b1, a1, raw_data, axis=0)
        x = signal.filtfilt(b2, a2, x, axis=0)
        x = signal.filtfilt(b3, a3, x, axis=0)
        x = signal.filtfilt(b4, a4, x, axis=0)
        x = signal.filtfilt(b5, a5, x, axis=0)
    else: # if 1D array
        x = signal.filtfilt(b1, a1, raw_data)
        x = signal.filtfilt(b2, a2, x)
        x = signal.filtfilt(b3, a3, x)
        x = signal.filtfilt(b4, a4, x)
        x = signal.filtfilt(b5, a5, x)
    return x

def create_text_json_from_mat(
    base_dir: Path, 
    output_json: Path, 
    prompt: str
) -> Path:
    """
    Scans a directory for .mat files, processes them, and creates a single
    JSON file with a format adapted from helper.py.

    Args:
        base_dir: The root directory for the dataset set (e.g., 'Resource/data/Train').
        output_json: The full path for the output JSON file.
        prompt: The human prompt to be included in the conversations.

    Returns:
        The path to the generated JSON file.
    """
    print(f"Processing data from: {base_dir}...")
    
    mat_files = sorted(list(base_dir.rglob("*.mat")))
    print(f"[info] Found {len(mat_files)} .mat files in {base_dir.name}.")

    if not mat_files:
        print(f"Warning: No .mat files found in {base_dir}")
        return None

    items = []
    for mat_file_path in tqdm(mat_files, desc=f"Processing {base_dir.name}"):
        try:
            # --- Adapt ID from helper.py ---
            rel_path_parts = mat_file_path.relative_to(base_dir).parts
            label = mat_file_path.stem
            
            # This is a simplification; adjust if folder structure is deeper/different
            subject = rel_path_parts[0] if len(rel_path_parts) > 2 else "unknown_subject"
            session = rel_path_parts[1] if len(rel_path_parts) > 2 else "unknown_session"
            item_id = f"{subject}_{session}_{label}"

            # --- Load and process EMG data ---
            mat_content = sio.loadmat(mat_file_path)
            raw_emg_data = mat_content["data"].astype(np.float64)
            filtered_emg_data = filter_emg_data(raw_emg_data)

            # Convert the numpy array to a string for the LLM
            emg_text_representation = ",".join(map(str, filtered_emg_data.flatten()))

            # --- Build item dictionary in the target format ---
            items.append({
                "id": item_id,
                "conversations": [
                    {"from": "human", "value": emg_text_representation},
                    {"from": "gpt", "value": label}, # Label from filename is the ground truth
                ],
            })
        except Exception as e:
            print(f"Could not process file {mat_file_path}: {e}")

    # --- Save to JSON ---
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
        
    print(f"[json] scanned_mat={len(mat_files)} saved={len(items)}")
    return output_json

def main():
    """
    Main function to parse arguments and run the conversion process for
    train, validation, and test sets.
    """
    parser = argparse.ArgumentParser(
        description="Convert EMG .mat files into a text-based JSON format for LLMs, "
                    "mimicking the format from helper.py."
    )
    parser.add_argument(
        "--base-data-dir", 
        type=Path, 
        default=Path("Resource/data"),
        help="Base directory containing Train, Val, and Test subdirectories."
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("Resource/data"),
        help="Directory to save the output JSON files."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Transcribe the sentence from the following EMG signal.",
        help="The 'human' prompt to use in the conversations."
    )
    args = parser.parse_args()

    # This follows the logic from helper.py:create_train_json to process splits
    split_names = ("Train", "Val", "Test")
    split_dirs = {name: args.base_data_dir / name for name in split_names}

    for name, split_dir in split_dirs.items():
        if not split_dir.is_dir():
            print(f"[json] skip {name}: not found -> {split_dir}")
            continue
        
        # Output filename will be train.json, val.json, etc.
        out_path = args.output_dir / f"{name.lower()}.json"

        saved_path = create_text_json_from_mat(
            base_dir=split_dir,
            output_json=out_path,
            prompt=args.prompt
        )
        if saved_path:
            print(f"Successfully saved: {saved_path}")
            
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()