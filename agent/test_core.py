"""Verify all core modules import cleanly."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test 1: Task Tree
from core.task_tree import TaskTree
tree = TaskTree("test-001", "http://target:5000")
tree.start_phase("recon")
tid = tree.add_task("recon", "Fingerprint target")
tree.start_task(tid)
tree.complete_task(tid, summary="Flask, PostgreSQL")
tree.complete_phase("recon", summary="Recon complete")
print("TaskTree: OK")
print(f"  Progress: {tree.get_overall_progress():.0f}%")
print(f"  Status: {tree.get_status_line()}")

# Test 2: Orchestrator import (just import, no run)
from core.orchestrator import Orchestrator, ScanCallbacks
from models.scan import ScanConfig
config = ScanConfig(target_url="http://localhost:5000")
orch = Orchestrator(config=config)
print(f"Orchestrator: OK (scan_id={orch.scan_id})")

# Test 3: All core modules together
from core.memory import ScanMemory
mem = ScanMemory(scan_id="test", config=config)
print("Memory: OK")

print("=" * 40)
print("ALL CORE MODULES VERIFIED")
