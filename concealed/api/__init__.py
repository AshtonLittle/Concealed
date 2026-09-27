"""Concealed Backend REST API.

Provides high-throughput real-time image obfuscation endpoints with
configurable adversarial parameters, conforming masks, and multi-format output.
"""

from concealed.api.app import app

__all__ = ["app"]
