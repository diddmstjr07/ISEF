#!/usr/bin/env python3
import sys
import argparse
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download  # [변경] 직접 import

def quote_cmd(cmd: list[str]) -> str:
    # 명령어 출력 가독성 향상
    return " ".join([f'"{c}"' if (" " in c or "\t" in c) else c for c in cmd])

def build_cmd(args) -> list[str]:
    # 실제 학습 코드(train_sft.py) 실행 명령어 생성
    cmd = [
        sys.executable, "-m", "module.src_text.train.train_sft",
        "--model_id", str(args.model_path),
        "--data_path", str(args.data_path),
        "--output_dir", str(args.output_dir),
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--dataloader_num_workers", str(args.dataloader_num_workers),
        
        # --- 고정 하이퍼파라미터 ---
        "--bf16", "True",
        "--lora_enable", "True",
        "--freeze_llm", "True",
        "--logging_steps", "10",
        "--save_steps", "500",
        "--save_total_limit", "2",
    ]

    # 선택적 인자 추가
    if args.num_train_epochs is not None:
        cmd += ["--num_train_epochs", str(args.num_train_epochs)]
    if args.learning_rate is not None:
        cmd += ["--learning-rate", str(args.learning_rate)]

    return cmd

def main():
    p = argparse.ArgumentParser()

    # --- 1. Model & Download Settings ---
    p.add_argument("--download-hf", action="store_true", help="Download from HF repo before training.")
    
    # 기본값 None -> 입력 없으면 물어봄
    p.add_argument("--hf-repo-id", default=None, help="HuggingFace Repository ID to download")
    p.add_argument("--hf-sha", default=None, help="Specific Commit SHA (Revision) for HF download")
    
    p.add_argument("--hf-download-dir", default="pretrained/ISEF", help="Directory to store HF download")
    p.add_argument("--model-path", default=None, help="Direct path to model (overwritten if download-hf is used)")

    # --- 2. Dataset & Output ---
    p.add_argument("--data-path", default="Resource/data/train.json")
    p.add_argument("--output-dir", default="output/text_only_model")

    # --- 3. Training Hyperparameters ---
    p.add_argument("--per-device-train-batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--dataloader-num-workers", type=int, default=16)
    p.add_argument("--num-train-epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-5)

    # --- 4. Misc ---
    p.add_argument("--dry-run", action="store_true", help="Print command only; do not execute")

    args = p.parse_args()

    # 경로 정규화
    args.data_path = Path(args.data_path).expanduser().resolve()
    args.output_dir = Path(args.output_dir).expanduser().resolve()

    # 데이터 경로 확인 (없으면 경고만 하고 진행하거나 에러 처리)
    if not args.data_path.exists():
         print(f"[Warning] Data path not found locally: {args.data_path}")

    # =========================================================
    # [수정됨] 모델 다운로드 로직 (huggingface_hub 직접 사용)
    # =========================================================
    if args.download_hf:
        # 1. Repo ID 입력
        if not args.hf_repo_id:
            default_repo = "diddmstjr/ISEF"
            user_repo = input(f"Target Model ID를 입력하세요 [Enter 입력 시 '{default_repo}']: ").strip()
            args.hf_repo_id = user_repo if user_repo else default_repo
        
        # 2. SHA Key 입력
        if not args.hf_sha:
            user_sha = input("SHA Key(Commit Hash)를 입력하세요 [없으면 Enter]: ").strip()
            args.hf_sha = user_sha if user_sha else None

        # 3. 다운로드 수행 (snapshot_download 사용)
        hf_dir = Path(args.hf_download_dir).expanduser().resolve()
        hf_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[HF] Downloading '{args.hf_repo_id}' (SHA: {args.hf_sha}) -> {hf_dir} ...")
        
        try:
            snapshot_download(
                repo_id=args.hf_repo_id,
                local_dir=str(hf_dir),
                revision=args.hf_sha,  # SHA Key 적용
                ignore_patterns=["*.msgpack", "*.h5"] # 불필요한 파일 제외 가능
            )
            print(f"[HF] Download complete.\n")
            args.model_path = hf_dir
            
        except Exception as e:
            print(f"\n[Error] Download failed: {e}")
            sys.exit(1)
    
    # =========================================================
    # 모델 경로 처리
    # =========================================================
    if not args.model_path:
        default_model = "Qwen/Qwen2-1.5B-Instruct"
        print(f"[Info] 모델 경로가 지정되지 않았습니다. 기본값을 사용합니다: {default_model}")
        args.model_path = default_model
    else:
        # 문자열이 경로 형식인 경우 절대 경로로 변환
        p_model = Path(args.model_path).expanduser()
        if p_model.exists():
            args.model_path = p_model.resolve()

    # 명령어 빌드 및 실행
    cmd = build_cmd(args)

    print("[Generated Command]")
    print(quote_cmd(cmd))
    print("-" * 60)

    if args.dry_run:
        return

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()