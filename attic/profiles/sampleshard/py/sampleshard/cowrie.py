"""
Cowrie codec shim for SampleShard.

Provides encode/decode that accept raw Python objects (dict, list, str, int,
etc.) and produce/consume Cowrie v3 binary format. Uses the cowrie-py package.
"""

from cowrie.gen2 import dumps, loads, MAGIC

encode = dumps
decode = loads
