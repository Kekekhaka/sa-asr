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

# Large Hub files download through hf_xet by default. It has hung with no timeout: at 0 bytes
# behind the corporate TLS proxy (its Rust TLS ignores the OS store above), and mid-file on
# Kaggle (a 2026-10-13 training run stalled before step 1 until the watchdog stopped it).
# Plain downloads time out and retry, so use them everywhere. The env var only counts if set
# before huggingface_hub is imported, which scripts may already have done: set the flag too.
import os as _os

_os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
try:
    import huggingface_hub.constants as _hf_constants

    _hf_constants.HF_HUB_DISABLE_XET = _os.environ["HF_HUB_DISABLE_XET"].lower() in ("1", "true", "yes", "on")
except ImportError:
    pass
