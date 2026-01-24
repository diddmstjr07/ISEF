#!/usr/bin/env python3
"""
Download a tar file from Google Drive using rclone, then extract it.
No interactive prompts: defaults are used automatically.
"""

import os
import sys
import subprocess
import tarfile
from pathlib import Path
import argparse


class RcloneGDrive:
    def __init__(self, remote_name: str = "gdrive"):
        self.remote_name = remote_name

    def check_rclone_installed(self) -> bool:
        """Check if rclone is installed"""
        result = subprocess.run(["which", "rclone"], capture_output=True, text=True)
        return result.returncode == 0

    def install_rclone(self) -> bool:
        """Install rclone (requires sudo in most environments)"""
        print("Installing rclone...")
        try:
            result = subprocess.run(
                "curl https://rclone.org/install.sh | sudo bash",
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                print("✓ rclone installation complete")
                return True
            print(f"✗ rclone installation failed: {result.stderr}")
            return False
        except Exception as e:
            print(f"✗ Error while installing rclone: {e}")
            return False

    def check_gdrive_configured(self) -> bool:
        """Check if the remote exists in rclone config"""
        result = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True)
        return f"{self.remote_name}:" in result.stdout

    def setup_gdrive_config(self) -> bool:
        """
        Start interactive rclone config. (OAuth cannot be fully automated reliably.)
        """
        print("\n=== Google Drive auth setup ===")
        print("Launching: rclone config")
        try:
            subprocess.run(["rclone", "config"])
            print("✓ Google Drive setup complete")
            return True
        except Exception as e:
            print(f"✗ Error during setup: {e}")
            return False

    def download_single_file(self, remote_path: str, local_file: str, show_progress: bool = True) -> bool:
        """
        Download a single file from remote to a local filename.
        Example:
          remote_path: "Resource.tar"  or "ISEF/full_corpus/Resource.tar"
          local_file:  "Resource.tar"
        """
        print(f"\nDownloading: {self.remote_name}:{remote_path} -> {local_file}")

        cmd = ["rclone", "copyto", f"{self.remote_name}:{remote_path}", local_file]
        if show_progress:
            cmd.append("-P")

        result = subprocess.run(cmd, text=True)
        if result.returncode == 0:
            print(f"✓ Download complete: {local_file}")
            return True

        print("✗ Download failed")
        return False

    def extract_tar_safe(self, tar_path: str, extract_to: str) -> bool:
        """
        Extract tar with basic path traversal protection.
        """
        print(f"\nExtracting: {tar_path} -> {extract_to}")

        try:
            os.makedirs(extract_to, exist_ok=True)

            def is_within_directory(base: str, target: str) -> bool:
                base_real = os.path.realpath(base)
                target_real = os.path.realpath(target)
                return target_real.startswith(base_real + os.sep) or target_real == base_real

            with tarfile.open(tar_path, "r:*") as tar:
                for m in tar.getmembers():
                    dest = os.path.join(extract_to, m.name)
                    if not is_within_directory(extract_to, dest):
                        raise RuntimeError(f"Unsafe path in tar: {m.name}")
                tar.extractall(path=extract_to)

            print("✓ Extraction complete")
            return True
        except Exception as e:
            print(f"✗ Error during extraction: {e}")
            return False


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-download and extract a tar from rclone Google Drive remote.")
    parser.add_argument("--remote", default="gdrive", help='rclone remote name (default: "gdrive")')
    parser.add_argument("--remote-path", default="Resource.tar", help='path inside remote (default: "Resource.tar")')
    parser.add_argument("--output-tar", default="Resource.tar", help='local tar filename (default: "Resource.tar")')
    parser.add_argument("--extract-to", default="./", help='extraction directory (default: "./")')
    parser.add_argument("--no-extract", action="store_true", help="download only; do not extract")
    return parser.parse_args()


def main() -> bool:
    args = parse_args()

    remote_path = args.remote_path
    output_tar = args.output_tar
    extract_to = args.extract_to

    print("\n" + "=" * 50)
    print("[*] Configuration Summary")
    print("=" * 50)
    print(f"Remote name:   {args.remote}")
    print(f"Remote path:   {remote_path}")
    print(f"Output tar:    {output_tar}")
    print(f"Extract to:    {extract_to}")
    print(f"Do extract?:   {not args.no_extract}")
    print("=" * 50)

    gdrive = RcloneGDrive(remote_name=args.remote)

    print("\n[1/4] Checking rclone...")
    if not gdrive.check_rclone_installed():
        print("[✗] rclone is not installed.")
        if not gdrive.install_rclone():
            return False
    else:
        print("[✓] rclone installed")

    print("\n[2/4] Checking Google Drive config...")
    if not gdrive.check_gdrive_configured():
        print("[✗] Google Drive remote not configured.")
        if not gdrive.setup_gdrive_config():
            return False
    else:
        print("[✓] Google Drive configured")

    print("\n[3/4] Downloading from Google Drive...")
    if not gdrive.download_single_file(remote_path, output_tar, show_progress=True):
        return False

    if args.no_extract:
        print("\n[4/4] Skipping extraction (--no-extract)")
        return True

    print("\n[4/4] Extracting tar...")
    if not gdrive.extract_tar_safe(output_tar, extract_to):
        return False

    print("\n" + "=" * 50)
    print("[✓] All tasks completed!")
    print("=" * 50)

    try:
        file_size = os.path.getsize(output_tar) / (1024**3)
        print(f"Tar file size: {file_size:.2f} GB")
    except Exception as e:
        print(f"Failed to check file size: {e}")

    return True
