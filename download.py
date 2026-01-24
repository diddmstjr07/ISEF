from huggingface_hub import snapshot_download

import time
import zipfile
import librosa
import numpy as np
import scipy.io as sio
from scipy import signal
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

import os
import base64
import getpass
import subprocess
from tqdm import tqdm
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from module.drive import main

class GithubDownload:
    @staticmethod
    def download(output_path: str="Resource/data"):
        os.makedirs(output_path, exist_ok=True)
        command = [
            'curl',
            '-L',
            '-o',
            f'{output_path}/corpus.json',
            'https://raw.githubusercontent.com/MML-Group/code4AVE-Speech/master/corpus.json'
        ]
        return subprocess.run(command)

class HuggingFaceDownload:
    def __init__(self, password: str | None = None, enc_path: str = "Resource/oJtYpLhVfD.enc"):
        if password is None:
            password = getpass.getpass("password: ")
        self.token = self.decrypt_file(file_path=enc_path, password=password)
        print(self.token)
        self.repo = "MML-Group/AVE-Speech"

    def derive_key(self, password: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def decrypt_file(self, file_path: str, password: str):
        with open(file_path, 'rb') as file:
            data = file.read()

        salt = data[:16]
        encrypted_data = data[16:]
        key = self.derive_key(password, salt)
        f = Fernet(key)
        decrypted_data = f.decrypt(encrypted_data)
        return decrypted_data.decode('utf-8')
    
    def download_part_dataset(
        self,
        subjects: tuple[int, ...] = (1,),
        modalities: tuple[str, ...] = ("EMG",),
        split: str = "Train",
        local_dir: str | Path = ".",
    ):
        allow_patterns = []
        for subject in subjects:
            for modality in modalities:
                allow_patterns.append(f"{split}/{modality}/subject_{subject}.zip")
        return snapshot_download(
            repo_id=self.repo,
            repo_type="dataset",
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
            token=self.token,
        )

    def dowload_full_dataset(
        self,
        local_dir: str | Path = ".",
    ):
        return snapshot_download(
            repo_id=self.repo,
            repo_type="dataset",
            local_dir=str(local_dir),
            token=self.token,
        )

class EMGPreprocessing:
    def __init__(self):
        self.fs = 1000

    def filter(self, raw_data):
        b1, a1 = signal.iirnotch(50, 30, self.fs)
        b2, a2 = signal.iirnotch(150, 30, self.fs)
        b3, a3 = signal.iirnotch(250, 30, self.fs)
        b4, a4 = signal.iirnotch(350, 30, self.fs)
        b5, a5 = signal.butter(4, [10 / (self.fs / 2), 400 / (self.fs / 2)], "bandpass")

        x = signal.filtfilt(b1, a1, raw_data, axis=1)
        x = signal.filtfilt(b2, a2, x, axis=1)
        x = signal.filtfilt(b3, a3, x, axis=1)
        x = signal.filtfilt(b4, a4, x, axis=1)
        x = signal.filtfilt(b5, a5, x, axis=1)
        return x

    def EMG_MFSC(self, x):
        x = x[:, 250:, :]
        n_mels = 36
        sr = 1000
        channel_list = []
        for j in range(x.shape[-1]):
            mfsc_x = np.zeros((x.shape[0], 36, n_mels))
            for i in range(x.shape[0]):
                norm_x = np.asfortranarray(x[i, :, j])
                tmp = librosa.feature.melspectrogram(
                    y=norm_x, sr=sr, n_mels=n_mels, n_fft=200, hop_length=50
                )
                tmp = librosa.power_to_db(tmp).T
                mfsc_x[i, :, :] = tmp

            mfsc_x = np.expand_dims(mfsc_x, axis=-1)
            channel_list.append(mfsc_x)
        data_x = np.concatenate(channel_list, axis=-1)
        mu = np.mean(data_x)
        std = np.std(data_x)
        data_x = (data_x - mu) / std
        data_x = data_x.transpose(0, 3, 1, 2)
        return data_x

    def load_and_preprocess_emg(self, mat_path: str):
        t_start = time.time()

        t0 = time.time()
        emg = sio.loadmat(mat_path)
        t1 = time.time()

        t2 = time.time()
        emg = np.expand_dims(emg["data"], axis=0)
        t3 = time.time()

        t4 = time.time()
        emg = self.filter(emg)
        t5 = time.time()

        t6 = time.time()
        emg = self.EMG_MFSC(emg)
        t7 = time.time()

        print(
            "[timing] loadmat={:.4f}s expand={:.4f}s filter={:.4f}s mfsc={:.4f}s total={:.4f}s".format(
                t1 - t0, t3 - t2, t5 - t4, t7 - t6, t7 - t_start
            )
        )
        return emg
    
class DatasetProcessor:
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)

    def unzip_files(self):
        zip_files = list(self.base_dir.rglob("*.zip"))
        print(f"\nFound {len(zip_files)} zip files. Starting unzip process...")

        for zip_path in tqdm(zip_files, desc="Unzipping", unit="file"):
            extract_to = zip_path.with_suffix('') 
            
            if extract_to.exists():
                continue
                
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            except zipfile.BadZipFile:
                tqdm.write(f"  Damaged: {zip_path.name}")

    def mat_static_directory(self):
        mat_files = list(self.base_dir.rglob("*.mat"))
        
        if not mat_files:
            return
        
        for path in mat_files:
            yield path
    
    def avi_static_directory(self):
        avi_files = list(self.base_dir.rglob("*.avi"))

        if not avi_files:
            return
        
        for path in avi_files:
            yield path

def list_subject_files(base_dir: Path, video_frame_count: int | None = 60) -> None:
    def save_stacked_channels_png(emg_tensor: np.ndarray, out_png: Path):
        specs = [emg_tensor[0, ch] for ch in range(emg_tensor.shape[1])]  # list of (T,F)
        stacked = np.concatenate(specs, axis=0)  # (C*T, F)
        
        plt.figure()
        plt.imshow(stacked, aspect="auto", origin="lower")
        plt.axis("off")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, bbox_inches="tight", pad_inches=0, dpi=200)
        plt.close()

    def center_crop(img: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
        crop_w, crop_h = crop_size
        h, w = img.shape[:2]
        left = max((w - crop_w) // 2, 0)
        top = max((h - crop_h) // 2, 0)
        return img[top : top + crop_h, left : left + crop_w]

    def convert_avi_to_mp4(
        avi_path: Path,
        out_path: Path,
        crop_size=(320, 240),
        resize=(88, 88),
    ) -> None:
        cap = cv2.VideoCapture(str(avi_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30

        writer = None
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if crop_size:
                    frame = center_crop(frame, crop_size)
                if resize:
                    frame = cv2.resize(frame, resize)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                if writer is None:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

                writer.write(frame)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
    
    out_root = base_dir.parent / f"{base_dir.name}"
    emgpreprocessing = EMGPreprocessing()

    processor = DatasetProcessor(base_dir=base_dir)
    processor.unzip_files()

    # Process EMG files
    mat_files = list(processor.mat_static_directory())
    for directory in tqdm(mat_files, desc="Converting EMG to Spectrogram", unit="file"):
        relative_path = directory.relative_to(base_dir)
        relative_path = [
            "EMG_IMG" if part == "EMG" else part 
            for part in relative_path.parts
        ]
        relative_path = Path(*relative_path)
        target_png_path = out_root / relative_path.with_suffix(".png")
        
        # Skip if already converted
        if target_png_path.exists():
            continue
        
        save_stacked_channels_png(emg_tensor=emgpreprocessing.load_and_preprocess_emg(mat_path=directory), out_png=target_png_path)
    
    # Process AVI files
    avi_files = list(processor.avi_static_directory())
    for directory in tqdm(avi_files, desc="Converting AVI to MP4", unit="file"):
        relative_path = directory.relative_to(base_dir)
        relative_path = [
            "Visual_MP4" if part == "Visual" else part
            for part in relative_path.parts
        ]
        relative_path = Path(*relative_path)
        target_mp4 = out_root / relative_path.with_suffix(".mp4")
        
        # Skip if already converted
        if target_mp4.exists():
            continue
        
        convert_avi_to_mp4(directory, target_mp4)

if __name__ == "__main__":
    # GithubDownload.download()
    # huggingfacedownload = HuggingFaceDownload()
    # huggingfacedownload.dowload_full_dataset(local_dir="Resource/data")
    # list_subject_files(Path("Resource/data"), video_frame_count=15)
    try:
        ok = main()
        sys.exit(0 if ok else 1)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
