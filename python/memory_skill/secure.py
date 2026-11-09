"""存储加密与篡改检测（AES-256-GCM）。

设计目标：挡得住 `cat` / `grep` / 备份外泄 / 机器上其它账号；**挡不住进程
本身**——插件要工作就得解密，所以密钥故意放在数据旁边（策略边界，不是
密码学边界，与参考实现口径一致）。

- 密文格式：``enc:v1:<base64(nonce|ct)>``，nonce 12 字节。
- GCM 自带完整性标签：解密失败 = 被改过。读取侧捕获异常后返回占位文本
  并计数，界面/注入层可以报"N 条通不过校验"。
- 旧数据（非 ``enc:v1:`` 前缀）原样读取，渐进迁移。
- 密钥：``<memory_dir>/master.key``（0600，首次自动生成）。
"""
from __future__ import annotations

import base64
import os
import secrets

_PREFIX = "enc:v1:"
_KEY: bytes | None = None
_KEY_FILE = "master.key"


def memory_dir(db_path: str | None = None) -> str:
    """密钥所在目录：优先显式 db_path 的目录，否则进程环境里的记忆目录。"""
    if db_path:
        return os.path.dirname(os.path.abspath(db_path))
    env = os.environ.get("MEMORY_SKILL_DB_PATH")
    if env:
        return os.path.dirname(os.path.abspath(env))
    return os.path.join(os.path.expanduser("~"), ".dsh", "memory")


def key_path() -> str:
    return os.path.join(memory_dir(), _KEY_FILE)


def _load_or_create_key() -> bytes:
    global _KEY
    if _KEY is not None:
        return _KEY
    path = key_path()
    try:
        with open(path, "rb") as f:
            raw = f.read().strip()
        if raw:
            _KEY = base64.b64decode(raw)
            return _KEY
    except FileNotFoundError:
        pass
    except Exception:
        pass
    key = secrets.token_bytes(32)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, base64.b64encode(key))
        finally:
            os.close(fd)
    except Exception:
        pass
    _KEY = key
    return key


def reset_key_cache() -> None:
    """测试/换密钥时用。"""
    global _KEY
    _KEY = None


def is_encrypted(value) -> bool:
    return isinstance(value, str) and value.startswith(_PREFIX)


def encrypt(text: str) -> str:
    """把明文加密成 ``enc:v1:...``；空值/已加密原样返回。"""
    if not text or is_encrypted(text):
        return text
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = secrets.token_bytes(12)
    ct = AESGCM(_load_or_create_key()).encrypt(nonce, text.encode("utf-8"), None)
    return _PREFIX + base64.b64encode(nonce + ct).decode("ascii")


class Corrupted(Exception):
    """密文被篡改 / 密钥不匹配。"""


def decrypt(text: str) -> str:
    """解密；旧明文原样返回；被篡改抛 ``Corrupted``。"""
    if not is_encrypted(text):
        return text
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    try:
        raw = base64.b64decode(text[len(_PREFIX):])
        nonce, ct = raw[:12], raw[12:]
        return AESGCM(_load_or_create_key()).decrypt(nonce, ct, None).decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        raise Corrupted(str(exc)) from exc


def decrypt_or_placeholder(text: str) -> tuple[str, bool]:
    """返回 (明文或占位, 是否完整)。读取路径统一走这里。"""
    try:
        return decrypt(text), True
    except Corrupted:
        return "[! 通不过完整性校验]", False


def encrypt_many(values):
    return [encrypt(v) for v in values]


__all__ = [
    "memory_dir", "key_path", "is_encrypted", "encrypt", "decrypt",
    "decrypt_or_placeholder", "Corrupted", "reset_key_cache",
]
