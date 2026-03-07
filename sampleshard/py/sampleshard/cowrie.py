"""
Cowrie codec shim for SampleShard.

Provides encode/decode that accept raw Python objects (dict, list, str, int,
etc.) and produce/consume Cowrie v2 binary format. This wraps the full
cowrie codec's from_any + Encoder and Decoder + to_native.
"""

import importlib.util
import os
import sys

# Load the full cowrie module from the sibling cowrie/python/ directory
# using importlib to avoid name collision with this module.
_cowrie_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "python", "cowrie.py")
)
_spec = importlib.util.spec_from_file_location("_cowrie_full", _cowrie_path)
_cowrie = importlib.util.module_from_spec(_spec)
# Register in sys.modules so dataclass introspection works
sys.modules["_cowrie_full"] = _cowrie
_spec.loader.exec_module(_cowrie)

MAGIC = _cowrie.MAGIC


def encode(obj):
    """Encode a Python object to Cowrie v2 binary bytes."""
    return _cowrie.dumps(obj)


def decode(data: bytes):
    """Decode Cowrie v2 binary bytes to a Python object."""
    return _cowrie.loads(data)
