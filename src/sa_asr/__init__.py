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
