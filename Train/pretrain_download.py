from huggingface_hub import snapshot_download

import base64
import getpass
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class HuggingFaceDownload:
    def __init__(
        self,
        repo: str = "diddmstjr/ISEF",
        password: str | None = None,
        enc_path: str | None = "Resource/oJtYpLhVfD.enc",
        token: str | None = None,
        repo_type: str = "model",  # <- dataset -> model
    ):
        self.repo = repo
        self.repo_type = repo_type

        # if token is not None:
        #     self.token = token
        # elif enc_path is not None:
        #     if password is None:
        #         password = getpass.getpass("password: ")
        #     self.token = self.decrypt_file(file_path=enc_path, password=password)
        # else:
        self.token = None  # public repo

    def derive_key(self, password: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def decrypt_file(self, file_path: str, password: str) -> str:
        with open(file_path, "rb") as file:
            data = file.read()
        salt = data[:16]
        encrypted_data = data[16:]
        key = self.derive_key(password, salt)
        f = Fernet(key)
        decrypted_data = f.decrypt(encrypted_data)
        return decrypted_data.decode("utf-8")

    def download_full_repo(self, local_dir: str | Path = "."):
        return snapshot_download(
            repo_id=self.repo,
            repo_type=self.repo_type,  # "model"
            local_dir=str(local_dir),
            token=self.token,
        )

    def download_by_patterns(
        self,
        local_dir: str | Path = ".",
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
    ):
        """
        allow_patterns ex) :
            ["*.json", "*.safetensors", "tokenizer/*", "processor/*"]
        """
        return snapshot_download(
            repo_id=self.repo,
            repo_type=self.repo_type,
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            token=self.token,
        )
