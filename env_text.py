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
    Applies a series of notch and bandpass filters.
    """
    fs = 1000
    # Create notch filters
    b1, a1 = signal.iirnotch(50, 30, fs)
    b2, a2 = signal.iirnotch(150, 30, fs)
    b3, a3 = signal.iirnotch(250, 30, fs)
    b4, a4 = signal.iirnotch(350, 30, fs)
    # Create Band-pass filter
    b5, a5 = signal.butter(4, [10 / (fs / 2), 400 / (fs / 2)], 'bandpass')

    # Apply filters
    if raw_data.ndim > 1:
        x = signal.filtfilt(b1, a1, raw_data, axis=0)
        x = signal.filtfilt(b2, a2, x, axis=0)
        x = signal.filtfilt(b3, a3, x, axis=0)
        x = signal.filtfilt(b4, a4, x, axis=0)
        x = signal.filtfilt(b5, a5, x, axis=0)
    else: 
        x = signal.filtfilt(b1, a1, raw_data)
        x = signal.filtfilt(b2, a2, x)
        x = signal.filtfilt(b3, a3, x)
        x = signal.filtfilt(b4, a4, x)
        x = signal.filtfilt(b5, a5, x)
    return x

def create_text_json_from_mat(
    base_dir: Path, 
    output_json: Path, 
    prompt: str,
    downsample_rate: int = 50,  # [수정] 50개 중 1개만 추출 (1000Hz -> 20Hz 효과)
    precision: int = 1          # [수정] 소수점 1자리까지만 저장
) -> Path:
    """
    Scans .mat files, downsamples/rounds data, and saves as JSON.
    """
    print(f"Processing data from: {base_dir}...")
    
    mat_files = sorted(list(base_dir.rglob("*.mat")))
    print(f"[info] Found {len(mat_files)} .mat files in {base_dir.name}.")

    if not mat_files:
        print(f"Warning: No .mat files found in {base_dir}")
        return None

    items = []
    
    # 토큰 길이 경고를 위한 대략적인 체크용
    max_len_warning_threshold = 6000 

    for mat_file_path in tqdm(mat_files, desc=f"Processing {base_dir.name}"):
        try:
            # --- ID Generation ---
            rel_path_parts = mat_file_path.relative_to(base_dir).parts
            label = mat_file_path.stem
            
            subject = rel_path_parts[0] if len(rel_path_parts) > 2 else "unknown_subject"
            session = rel_path_parts[1] if len(rel_path_parts) > 2 else "unknown_session"
            item_id = f"{subject}_{session}_{label}"

            # --- Load and Process EMG ---
            mat_content = sio.loadmat(mat_file_path)
            raw_emg_data = mat_content["data"].astype(np.float64)
            filtered_emg_data = filter_emg_data(raw_emg_data)

            # [핵심 수정 1] 다운샘플링 (Downsampling)
            # ::downsample_rate 슬라이싱을 사용하여 데이터 길이를 줄임
            # 예: 1000개의 샘플 -> 100개로 축소
            downsampled_data = filtered_emg_data[::downsample_rate]

            # [핵심 수정 2] 소수점 자르기 및 문자열 변환
            # flatten()으로 1차원으로 편 후, f-string으로 소수점 자리수 제한
            flat_data = downsampled_data.flatten()
            emg_str_list = [f"{x:.{precision}f}" for x in flat_data]
            emg_text_representation = ", ".join(emg_str_list)

            # [핵심 수정 3] 프롬프트와 데이터 결합
            # 단순히 데이터만 넣지 않고, 지시문(prompt)과 함께 포맷팅
            final_user_input = f"{prompt}\n\nEMG Sequence: [{emg_text_representation}]"

            # [추가] 길이가 너무 길면 Truncation (학습 속도 저하 방지)
            # 대략 1토큰 ≈ 3~4글자. 
            if len(final_user_input) > max_len_warning_threshold * 3:
                # 너무 긴 경우 뒤쪽 데이터를 잘라냄 (주의: 시계열 특성상 뒤가 잘리는게 맞는지 고려 필요)
                # 여기서는 단순히 문자열 길이로 자름
                final_user_input = final_user_input[:max_len_warning_threshold * 3] + "...]"

            # --- Build Item ---
            items.append({
                "id": item_id,
                "conversations": [
                    {"from": "user", "value": final_user_input},  # [수정] human -> user (Qwen 포맷)
                    {"from": "assistant", "value": label},        # [수정] gpt -> assistant (Qwen 포맷)
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
    parser = argparse.ArgumentParser(description="Convert EMG .mat to text-JSON for LLM.")
    parser.add_argument("--base-data-dir", type=Path, default=Path("Resource/data"), help="Base data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("Resource/data"), help="Output directory")
    parser.add_argument("--prompt", type=str, default="Analyze the patterns in the following EMG signal sequence and predict the gesture label.", help="Instruction prompt")
    
    # [추가 옵션] 압축 강도 조절
    parser.add_argument("--downsample", type=int, default=50, help="Downsample rate (default: 50, keep 1 out of 50)")
    parser.add_argument("--precision", type=int, default=1, help="Floating point precision (default: 1)")

    args = parser.parse_args()

    split_names = ("Train", "Val", "Test")
    split_dirs = {name: args.base_data_dir / name for name in split_names}

    for name, split_dir in split_dirs.items():
        if not split_dir.is_dir():
            print(f"[json] skip {name}: not found -> {split_dir}")
            continue
        
        out_path = args.output_dir / f"{name.lower()}.json"

        create_text_json_from_mat(
            base_dir=split_dir,
            output_json=out_path,
            prompt=args.prompt,
            downsample_rate=args.downsample,
            precision=args.precision
        )

    print("\nProcessing complete. Don't forget to re-run your training script!")

if __name__ == "__main__":
    main()