"""Multilingual speech-to-text for South African languages."""

__version__ = "0.1.0"

# Corporate networks that re-sign HTTPS traffic break Python's bundled CA list
# (CERTIFICATE_VERIFY_FAILED when downloading from the Hugging Face Hub). truststore makes
# Python use the operating system's certificate store instead, like browsers and curl do.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

# Large Hub files download through hf_xet, whose own (Rust) TLS ignores the OS store above and
# stalls at 0 bytes behind such a proxy. On the Windows laptop, use plain Python downloads instead.
import os as _os
import sys as _sys

if _sys.platform == "win32":
    _os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
