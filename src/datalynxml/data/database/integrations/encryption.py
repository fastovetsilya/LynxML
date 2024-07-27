import base64
import hashlib
import os
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes


def encrypt(raw, key):
    key = hashlib.sha256(key.encode("utf-8")).digest()

    BS = AES.block_size
    pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS)

    raw = base64.b64encode(pad(raw).encode("utf8"))
    iv = get_random_bytes(AES.block_size)
    cipher = AES.new(key=key, mode=AES.MODE_CFB, iv=iv)
    return base64.b64encode(iv + cipher.encrypt(raw))


def decrypt(enc, key):
    key = hashlib.sha256(key.encode("utf-8")).digest()

    unpad = lambda s: s[: -ord(s[-1:])]

    enc = base64.b64decode(enc)
    iv = enc[: AES.block_size]
    cipher = AES.new(key, AES.MODE_CFB, iv)
    return unpad(base64.b64decode(cipher.decrypt(enc[AES.block_size :])).decode("utf8"))


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    text = input()
    print(text)
    encrypted_text = encrypt(text, key=os.getenv("ENCRYPTION_KEY"))
    print(encrypted_text.decode())
    decrypted_text = decrypt(encrypted_text, key=os.getenv("ENCRYPTION_KEY"))
    print(decrypted_text)
