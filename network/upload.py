import os
import zipfile
import base64
import getpass
from tqdm import tqdm
from tqdm.utils import CallbackIOWrapper
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from huggingface_hub import HfApi

class HuggingFaceUpload:
    def __init__(self, password: str | None = None, enc_path: str = "Resource/oJtYpLhVfD.enc"):
        if password is None:
            password = getpass.getpass("password: ")
        self.token = self.decrypt_file(file_path=enc_path, password=password)
        self.api = HfApi(token=self.token)

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

    def upload_file(self, file_path: str, repo_id: str, path_in_repo: str, repo_type: str = "dataset"):
        file_size = os.path.getsize(file_path)
        with open(file_path, "rb") as f, tqdm(
            total=file_size, unit="B", unit_scale=True, desc=f"Uploading {os.path.basename(file_path)}"
        ) as pbar:
            wrapped_file = CallbackIOWrapper(pbar.update, f, "read")
            self.api.upload_file(
                path_or_fileobj=wrapped_file,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
            )

def zip_dir_with_progress(dir_path, zip_file):
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_list.append(os.path.join(root, file))

    with tqdm(total=len(file_list), unit="file", desc=f"Compressing {os.path.basename(dir_path)}") as pbar:
        for file in file_list:
            archive_path = os.path.relpath(file, os.path.dirname(dir_path))
            zip_file.write(file, archive_path)
            pbar.update(1)

def main():
    source_base_dir = 'Resource/data'
    dirs_to_zip_and_upload = ['Train', 'Test', 'Val']

    repo_id = input("Enter the Hugging Face repository ID (e.g., username/repo_name): ")
    hf_uploader = HuggingFaceUpload()

    for dir_name in dirs_to_zip_and_upload:
        dir_to_zip = os.path.join(source_base_dir, dir_name)
        output_zip_file = f"{dir_to_zip}.zip"

        if not os.path.isdir(dir_to_zip):
            print(f"Directory {dir_to_zip} not found. Skipping.")
            continue

        # 1. Compress the directory with progress
        with zipfile.ZipFile(output_zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zip_dir_with_progress(dir_to_zip, zipf)

        # 2. Upload the compressed file with progress
        path_in_repo = os.path.join('data', f"{dir_name}.zip")
        hf_uploader.upload_file(
            file_path=output_zip_file,
            repo_id=repo_id,
            path_in_repo=path_in_repo
        )
        print(f"Successfully uploaded {output_zip_file} to {repo_id}")
        
        # 3. Optional: remove the created zip file
        # os.remove(output_zip_file)


if __name__ == '__main__':
    main()