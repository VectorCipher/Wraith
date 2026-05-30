"""
WRAITH Task Tree

Manages the hierarchical progress tree for a penetration test scan.
Built on top of the TaskNode data model, this module provides the
high-level API the orchestrator uses to create phases, track tasks,
and generate progress context for LLM prompts.

The tree structure mirrors a real pentest workflow:

    Root: Scan → http://target:5000
    ├── Phase: Initialization       [COMPLETED ✓]
    │   ├── Task: Connect scanner   [COMPLETED]
    │   └── Task: Verify models     [COMPLETED]
    ├── Phase: Reconnaissance       [COMPLETED ✓]
    │   ├── Task: Fingerprint       [COMPLETED] → Flask, PostgreSQL
    │   ├── Task: Crawl             [COMPLETED] → 15 endpoints
    │   └── Task: Code analysis     [COMPLETED] → 8 findings
    ├── Phase: Exploitation         [IN PROGRESS ⟳]
    │   ├── Task: SQLi /api/login   [COMPLETED] → VULNERABLE
    │   ├── Task: XSS /search       [IN PROGRESS]
    │   └── Task: IDOR /api/users   [PENDING]
    └── Phase: Reporting            [PENDING ○]
"""

from __future__ import annotations

from datetime import datetime

from models.task import TaskNode, TaskType, TaskStatus
from utils.logger import get_logger

logger = get_logger("core.task_tree")


class TaskTree:
    """
    Manages the scan progress tree.

    Provides a clean API for the orchestrator to:
    - Create the standard scan phases
    - Add tasks as the AI decides what to do
    - Track progress in real time
    - Generate context strings for LLM prompts and CLI display

    Usage:
        tree = TaskTree(scan_id="wraith-001", target_url="http://target:5000")
        tree.start_phase("recon")
        task_id = tree.add_task("recon", "Fingerprint target")
        tree.start_task(task_id)
        tree.complete_task(task_id, summary="Flask, PostgreSQL detected")
        tree.complete_phase("recon")
    """

    # Standard phase definitions for every scan
    PHASES = [
        ("init", "Initialization"),
        ("recon", "Reconnaissance"),
        ("analysis", "Analysis"),
        ("exploitation", "Exploitation"),
        ("post_exploit", "Post-Exploitation"),
        ("reporting", "Reporting"),
    ]

    def __init__(self, scan_id: str, target_url: str) -> None:
        # Create root node
        self._root = TaskNode(
            id=f"root-{scan_id}",
            name=f"Scan → {target_url}",
            task_type=TaskType.ROOT,
        )

        # Create standard phases
        self._phases: dict[str, TaskNode] = {}
        for phase_id, phase_name in self.PHASES:
            phase_node = TaskNode(
                id=f"phase-{phase_id}",
                name=phase_name,
                task_type=TaskType.PHASE,
            )
            self._root.add_child(phase_node)
            self._phases[phase_id] = phase_node

        # Fast lookup index: task_id → TaskNode
        self._task_index: dict[str, TaskNode] = {}
        self._task_counter: int = 0

        # Current active phase
        self._current_phase: str | None = None

        logger.info(f"Task tree created — {len(self.PHASES)} phases")

    # ===================================================================
    # Properties
    # ===================================================================

    @property
    def root(self) -> TaskNode:
        return self._root

    @property
    def current_phase(self) -> str | None:
        return self._current_phase

    @property
    def current_phase_node(self) -> TaskNode | None:
        if self._current_phase:
            return self._phases.get(self._current_phase)
        return None

    # ===================================================================
    # Phase Management
    # ===================================================================

    def start_phase(self, phase_id: str) -> None:
        """Mark a phase as in-progress."""
        phase = self._phases.get(phase_id)
        if not phase:
            logger.warning(f"Unknown phase: {phase_id}")
            return

        phase.mark_in_progress()
        self._current_phase = phase_id
        logger.info(f"Phase started: {phase.name}")

    def complete_phase(self, phase_id: str, summary: str = "") -> None:
        """Mark a phase as completed with optional summary."""
        phase = self._phases.get(phase_id)
        if not phase:
            return

        # Count findings from child tasks
        total_findings = sum(c.findings_count for c in phase.children)
        phase.findings_count = total_findings
        phase.mark_completed(summary=summary)

        if self._current_phase == phase_id:
            self._current_phase = None

        logger.info(
            f"Phase completed: {phase.name} "
            f"({len(phase.children)} tasks, {total_findings} findings)"
        )

    def fail_phase(self, phase_id: str, reason: str = "") -> None:
        """Mark a phase as failed."""
        phase = self._phases.get(phase_id)
        if phase:
            phase.mark_failed(reason=reason)
            logger.warning(f"Phase failed: {phase.name} — {reason}")

    def skip_phase(self, phase_id: str, reason: str = "") -> None:
        """Mark a phase as skipped."""
        phase = self._phases.get(phase_id)
        if phase:
            phase.mark_skipped(reason=reason)
            logger.info(f"Phase skipped: {phase.name} — {reason}")

    # ===================================================================
    # Task Management
    # ===================================================================

    def add_task(
        self,
        phase_id: str,
        name: str,
        description: str = "",
    ) -> str:
        """
        Add a new task to a phase.
        Returns the generated task_id for later reference.
        """
        phase = self._phases.get(phase_id)
        if not phase:
            logger.warning(f"Cannot add task — unknown phase: {phase_id}")
            return ""

        self._task_counter += 1
        task_id = f"task-{self._task_counter:03d}"

        task = TaskNode(
            id=task_id,
            name=name,
            task_type=TaskType.TASK,
            description=description,
        )

        phase.add_child(task)
        self._task_index[task_id] = task

        logger.debug(f"Task added: [{phase_id}] {name} ({task_id})")
        return task_id

    def start_task(self, task_id: str) -> None:
        """Mark a task as in-progress."""
        task = self._task_index.get(task_id)
        if task:
            task.mark_in_progress()
            logger.debug(f"Task started: {task.name}")

    def complete_task(
        self,
        task_id: str,
        summary: str = "",
        findings_count: int = 0,
    ) -> None:
        """Mark a task as completed."""
        task = self._task_index.get(task_id)
        if task:
            task.findings_count = findings_count
            task.mark_completed(summary=summary)
            logger.debug(f"Task completed: {task.name} — {summary}")

    def fail_task(self, task_id: str, reason: str = "") -> None:
        """Mark a task as failed."""
        task = self._task_index.get(task_id)
        if task:
            task.mark_failed(reason=reason)
            logger.debug(f"Task failed: {task.name} — {reason}")

    def skip_task(self, task_id: str, reason: str = "") -> None:
        """Mark a task as skipped."""
        task = self._task_index.get(task_id)
        if task:
            task.mark_skipped(reason=reason)

    # ===================================================================
    # Subtask Management
    # ===================================================================

    def add_subtask(
        self,
        parent_task_id: str,
        name: str,
        description: str = "",
    ) -> str:
        """Add a subtask under an existing task."""
        parent = self._task_index.get(parent_task_id)
        if not parent:
            return ""

        self._task_counter += 1
        subtask_id = f"subtask-{self._task_counter:03d}"

        subtask = TaskNode(
            id=subtask_id,
            name=name,
            task_type=TaskType.SUBTASK,
            description=description,
        )

        parent.add_child(subtask)
        self._task_index[subtask_id] = subtask
        return subtask_id

    def complete_subtask(
        self,
        subtask_id: str,
        summary: str = "",
        findings_count: int = 0,
    ) -> None:
        """Mark a subtask as completed."""
        self.complete_task(subtask_id, summary=summary, findings_count=findings_count)

    # ===================================================================
    # Progress Queries
    # ===================================================================

    def get_overall_progress(self) -> float:
        """Get scan-wide progress as a percentage (0-100)."""
        return self._root.progress_percent

    def get_phase_progress(self, phase_id: str) -> float:
        """Get progress for a specific phase."""
        phase = self._phases.get(phase_id)
        return phase.progress_percent if phase else 0.0

    def get_active_tasks(self) -> list[TaskNode]:
        """Get all currently in-progress tasks across all phases."""
        active = []
        for phase in self._phases.values():
            for task in phase.children:
                if task.is_active:
                    active.append(task)
                for sub in task.children:
                    if sub.is_active:
                        active.append(sub)
        return active

    def get_pending_tasks(self, phase_id: str | None = None) -> list[TaskNode]:
        """Get pending tasks, optionally for a specific phase."""
        phases = [self._phases[phase_id]] if phase_id else self._phases.values()
        pending = []
        for phase in phases:
            for task in phase.children:
                if task.status == TaskStatus.PENDING:
                    pending.append(task)
        return pending

    def get_total_findings(self) -> int:
        """Get total findings count across all tasks."""
        total = 0
        for phase in self._phases.values():
            for task in phase.children:
                total += task.findings_count
                for sub in task.children:
                    total += sub.findings_count
        return total

    # ===================================================================
    # Context for LLM Prompts
    # ===================================================================

    def build_progress_context(self) -> str:
        """
        Build a context string showing scan progress for LLM prompts.
        The AI uses this to understand what's been done and what's left.
        """
        parts = [
            "## Scan Progress",
            f"Overall: {self.get_overall_progress():.0f}% complete",
            "",
        ]

        for phase_id, _ in self.PHASES:
            phase = self._phases[phase_id]
            line = f"- {phase.status_icon} **{phase.name}** [{phase.status.value}]"

            if phase.result_summary:
                line += f" — {phase.result_summary}"

            if phase.children:
                done = phase.completed_children_count
                total = phase.child_count
                if total > 0:
                    line += f" ({done}/{total} tasks)"

            parts.append(line)

            # Show active and recently completed tasks
            for task in phase.children:
                if task.is_active or (task.is_done and task.findings_count > 0):
                    indent = "  "
                    task_line = f"{indent}{task.status_icon} {task.name}"
                    if task.result_summary:
                        task_line += f" → {task.result_summary}"
                    parts.append(task_line)

        return "\n".join(parts)

    def build_current_phase_context(self) -> str:
        """Build detailed context for the current active phase."""
        if not self._current_phase:
            return ""

        phase = self._phases[self._current_phase]
        parts = [
            f"## Current Phase: {phase.name}",
            f"Progress: {phase.progress_percent:.0f}%",
            "",
        ]

        for task in phase.children:
            line = f"- {task.status_icon} {task.name}"
            if task.result_summary:
                line += f" → {task.result_summary}"
            parts.append(line)

            for sub in task.children:
                sub_line = f"  {sub.status_icon} {sub.name}"
                if sub.result_summary:
                    sub_line += f" → {sub.result_summary}"
                parts.append(sub_line)

        return "\n".join(parts)

    # ===================================================================
    # Display
    # ===================================================================

    def to_display_string(self) -> str:
        """Full tree as a formatted string for CLI display."""
        return self._root.to_display_string()

    def get_status_line(self) -> str:
        """One-line status summary for CLI progress bar."""
        active = self.get_active_tasks()
        progress = self.get_overall_progress()

        if active:
            current = active[0].name
            return f"[{progress:.0f}%] {current}"

        if progress >= 100:
            return f"[{progress:.0f}%] Scan complete"

        return f"[{progress:.0f}%] Waiting..."
