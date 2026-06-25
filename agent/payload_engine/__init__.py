"""
WRAITH Payload Engine Package (v2)

Mutation and logging system for payloads. When a payload fails,
the mutation engine suggests alternatives. All results are logged
for statistical analysis and strategy optimization.

Components:
    - PayloadLogger:  Records every payload attempt for analysis
    - PayloadMutator: Generates payload mutations to bypass WAFs/filters
"""

from payload_engine.logger import PayloadLogger
from payload_engine.mutator import PayloadMutator

__all__ = ["PayloadLogger", "PayloadMutator"]
