"""
WRAITH Task Tree Data Models


  Root: Scan → http://localhost:5000
  ├── Phase: Reconnaissance [COMPLETED ✓]
  │   ├── Task: Technology Detection [COMPLETED] → Flask, PostgreSQL
  │   ├── Task: Source Code Analysis [COMPLETED] → 8 findings
  │   └── Task: Endpoint Mapping [COMPLETED] → 15 endpoints
  ├── Phase: Exploitation [IN PROGRESS ⟳]
  │   ├── Task: SQLi on POST /api/login [COMPLETED] → VULNERABLE
  │   │   └── Subtask: Data extraction [COMPLETED] → dumped users table
  │   ├── Task: XSS on GET /search [IN PROGRESS]
  │   └── Task: IDOR on GET /api/users/{id} [PENDING]
  └── Phase: Reporting [PENDING ○]

Models:
  TaskType   → What level (root/phase/task/subtask)
  TaskStatus → Current state (pending/in_progress/completed/failed/skipped)
  TaskNode   → A single node in the tree (has children for nesting)

The orchestrator builds and updates this tree as the scan runs.
The CLI renders it for real-time visual progress.
The AI uses it for context — "what have we done, what's left?"

Usage:
    root = TaskNode(
        id="root",
        name="Scan http://localhost:5000",
        task_type=TaskType.ROOT,
    )
    recon = TaskNode(
        id="phase-recon",
        name="Reconnaissance",
        task_type=TaskType.PHASE,
    )
    root.add_child(recon)
    recon.mark_in_progress()
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class TaskType(str, Enum):
    """
    Level of a node in the task tree hierarchy.

    ROOT:     Top-level node. Only one per scan.
              "Scan → http://localhost:5000"

    PHASE:    Major scan phase. Children of root.
              "Reconnaissance", "Exploitation", "Reporting"

    TASK:     Individual test action. Children of phases.
              "SQLi on /api/login", "XSS on /search"

    SUBTASK:  Granular step within a task. Children of tasks.
              "Data extraction", "WAF bypass attempt"

    Hierarchy:
      ROOT
      └── PHASE
          └── TASK
              └── SUBTASK
    """

    ROOT = "root"
    PHASE = "phase"
    TASK = "task"
    SUBTASK = "subtask"


class TaskStatus(str, Enum):
    """
    Current state of a task node.

    PENDING:      Not started yet. Waiting in queue.
    IN_PROGRESS:  Currently being executed.
    COMPLETED:    Finished successfully.
    FAILED:       Attempted but encountered an error.
    SKIPPED:      Deliberately skipped (e.g., attack disabled in config).

    State transitions:
      PENDING → IN_PROGRESS → COMPLETED
                            → FAILED
      PENDING → SKIPPED
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskNode(BaseModel):
    """
    A single node in the task tree.

    Each node represents a piece of work in the penetration test.
    Nodes can have children, creating the hierarchical tree structure.

    The orchestrator creates and updates nodes as the scan progresses.
    The CLI renders the tree for visual progress tracking.
    The AI receives a text representation of the tree in its prompts
    so it knows what's been done and what's remaining.

    Key behaviors:
    - add_child(): Attach a child node (phase, task, or subtask)
    - mark_in_progress(): Set status and record start time
    - mark_completed(): Set status, record end time, add summary
    - mark_failed(): Set status with failure reason
    - mark_skipped(): Set status with skip reason
    """

    id: str
    name: str
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    description: str = ""
    result_summary: str = ""
    findings_count: int = 0
    children: list["TaskNode"] = []
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def add_child(self, child: "TaskNode") -> None:
        """
        Add a child node to this node.

        Example:
            recon_phase.add_child(port_scan_task)
            recon_phase.add_child(code_analysis_task)
        """
        self.children.append(child)

    def mark_in_progress(self) -> None:
        """
        Mark this task as currently running.
        Records the start timestamp.
        """
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()

    def mark_completed(self, summary: str = "") -> None:
        """
        Mark this task as successfully completed.
        Records end timestamp and optional result summary.

        Args:
            summary: Brief description of what was found/achieved.
                     Examples: "Found 3 open ports"
                              "VULNERABLE — SQL Injection confirmed"
                              "15 endpoints discovered"
        """
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        if summary:
            self.result_summary = summary

    def mark_failed(self, reason: str = "") -> None:
        """
        Mark this task as failed.
        Records end timestamp and failure reason.

        Args:
            reason: Why it failed.
                    Examples: "Target unreachable"
                             "Scanner timeout"
        """
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.result_summary = reason

    def mark_skipped(self, reason: str = "") -> None:
        """
        Mark this task as deliberately skipped.

        Args:
            reason: Why it was skipped.
                    Examples: "Attack type disabled in config"
                             "No source code provided"
        """
        self.status = TaskStatus.SKIPPED
        self.result_summary = reason

    @property
    def duration_seconds(self) -> float | None:
        """Calculate task duration. Returns None if not started."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def is_active(self) -> bool:
        """Check if this task is currently running."""
        return self.status == TaskStatus.IN_PROGRESS

    @property
    def is_done(self) -> bool:
        """Check if this task is finished (completed, failed, or skipped)."""
        return self.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.SKIPPED,
        )

    @property
    def child_count(self) -> int:
        """Number of direct children."""
        return len(self.children)

    @property
    def completed_children_count(self) -> int:
        """Number of children that are done."""
        return sum(1 for c in self.children if c.is_done)

    @property
    def progress_percent(self) -> float:
        """Percentage of children completed. 0 if no children."""
        if not self.children:
            return 100.0 if self.is_done else 0.0
        return (self.completed_children_count / self.child_count) * 100

    @property
    def status_icon(self) -> str:
        """Icon representing current status for CLI display."""
        icons = {
            TaskStatus.PENDING: "○",
            TaskStatus.IN_PROGRESS: "⟳",
            TaskStatus.COMPLETED: "✓",
            TaskStatus.FAILED: "✗",
            TaskStatus.SKIPPED: "⊘",
        }
        return icons.get(self.status, "?")

    def to_display_string(self, indent: int = 0) -> str:
        """
        Generate a text representation of this node and all children.
        Used for CLI display and LLM context prompts.

        Example output:
          ✓ Reconnaissance [COMPLETED]
            ✓ Port Scan [COMPLETED] → Found 3 open ports
            ✓ Code Analysis [COMPLETED] → 8 findings
            ⟳ Endpoint Mapping [IN PROGRESS]
        """
        prefix = "  " * indent
        status_str = self.status.value.upper()
        line = f"{prefix}{self.status_icon} {self.name} [{status_str}]"

        if self.result_summary:
            line += f" → {self.result_summary}"

        if self.findings_count > 0:
            line += f" ({self.findings_count} findings)"

        lines = [line]

        for child in self.children:
            lines.append(child.to_display_string(indent + 1))

        return "\n".join(lines)