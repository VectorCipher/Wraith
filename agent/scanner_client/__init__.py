"""
WRAITH Scanner Client Package

Async gRPC client for communicating with the Go scanner engine.
This package is the ONLY Python code that talks to the Go scanner.

Usage:
    from scanner_client import ScannerClient

    client = ScannerClient()
    await client.connect()
    tech = await client.fingerprint_target("http://target:5000")
    await client.close()
"""

from scanner_client.client import ScannerClient, ScannerStatus

__all__ = [
    "ScannerClient",
    "ScannerStatus",
]
