import asyncio
from datetime import datetime

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table
from rich.text import Text

from core.orchestrator import ScanCallbacks
from models.task import TaskNode, TaskStatus
from models.scan import Vulnerability

class LiveDashboard(ScanCallbacks):
    """
    Live rich dashboard for the terminal.
    Implements the ScanCallbacks interface from the Orchestrator.
    """

    def __init__(self, target_url: str, db_manager=None, scan_id: str = None):
        self.console = Console()
        self.target_url = target_url
        self.start_time = datetime.now()
        self.db = db_manager
        self.scan_id = scan_id
        
        # State
        self.current_phase = "Initializing"
        self.active_tasks: dict[str, str] = {}
        self.completed_tasks: list[str] = []
        self.reasoning_log: list[str] = []
        self.vuln_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
        self.requests_sent = 0
        self.endpoints_tested = 0
        self.total_endpoints = 0
        self.scan_progress = 0.0

        # UI Components
        self.layout = self._make_layout()
        self.live = Live(self.layout, console=self.console, refresh_per_second=4)

    def _make_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout(name="root")
        layout.split(
            Layout(name="header", size=3),
            Layout(name="progress", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5)
        )
        layout["main"].split_row(
            Layout(name="tasks", ratio=1),
            Layout(name="reasoning", ratio=2)
        )
        return layout

    def _generate_header(self) -> Panel:
        """Generate the header panel."""
        table = Table.grid(expand=True)
        table.add_column(justify="left", style="cyan", no_wrap=True)
        table.add_column(justify="right", style="magenta", no_wrap=True)
        table.add_row(
            "WRAITH AUTONOMOUS PENTESTER",
            f"Target: {self.target_url}"
        )
        return Panel(table, style="bold white on blue")

    def _generate_progress(self) -> Panel:
        """Generate the overall progress bar."""
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=True
        )
        progress.add_task("Scan Progress", total=100, completed=self.scan_progress)
        return Panel(progress, border_style="green")

    def _generate_tasks_panel(self) -> Panel:
        """Generate the tasks panel."""
        table = Table(show_header=False, expand=True, box=None)
        table.add_column("Status", style="bold", width=3)
        table.add_column("Task")

        # Show a few recent completed tasks
        for task in self.completed_tasks[-3:]:
            table.add_row("[green][✓][/green]", task)

        # Show active tasks
        for task_id, task_name in self.active_tasks.items():
            table.add_row("[yellow][⟳][/yellow]", f"{task_name}...")

        return Panel(
            table, 
            title=f"Active Phase: {self.current_phase.capitalize()}", 
            border_style="cyan"
        )

    def _generate_reasoning_panel(self) -> Panel:
        """Generate the AI reasoning log panel."""
        text = Text()
        for log in self.reasoning_log[-10:]:
            text.append(log + "\n")
        return Panel(text, title="Live AI Reasoning", border_style="magenta")

    def _generate_footer(self) -> Panel:
        """Generate the stats footer."""
        duration = datetime.now() - self.start_time
        duration_str = str(duration).split('.')[0]
        
        stats = Table.grid(expand=True, padding=(0, 2))
        stats.add_column(justify="left", ratio=1)
        stats.add_column(justify="center", ratio=1)
        stats.add_column(justify="right", ratio=1)
        
        endpoints_str = f"{self.endpoints_tested}/{self.total_endpoints}" if self.total_endpoints else f"{self.endpoints_tested}"
        
        stats.add_row(
            f"Endpoints: {endpoints_str} tested",
            f"Requests: {self.requests_sent:,}",
            f"Duration: {duration_str}"
        )
        
        vuln_str = (
            f"[red]{self.vuln_counts['critical']} Critical[/red], "
            f"[yellow]{self.vuln_counts['high']} High[/yellow], "
            f"[cyan]{self.vuln_counts['medium']} Medium[/cyan]"
        )
        stats.add_row(f"Vulnerabilities: {vuln_str}", "", "")
        
        return Panel(stats, title="Scan Stats", border_style="blue")

    def update_layout(self):
        """Update all dynamic sections of the layout."""
        self.layout["header"].update(self._generate_header())
        self.layout["progress"].update(self._generate_progress())
        self.layout["tasks"].update(self._generate_tasks_panel())
        self.layout["reasoning"].update(self._generate_reasoning_panel())
        self.layout["footer"].update(self._generate_footer())

    def start(self):
        """Start the live display."""
        self.live.start()

    def stop(self):
        """Stop the live display."""
        self.live.stop()

    # ===================================================================
    # ScanCallbacks Implementation
    # ===================================================================

    async def on_scan_start(self, target_url: str) -> None:
        self.update_layout()

    async def on_scan_complete(self, summary: str) -> None:
        self.scan_progress = 100.0
        self.update_layout()

    async def on_phase_start(self, phase_name: str, description: str = "") -> None:
        self.current_phase = phase_name
        self.update_layout()

    async def on_phase_complete(self, phase_name: str, summary: str) -> None:
        self.update_layout()

    async def on_task_start(self, task_id: str, task_name: str) -> None:
        self.active_tasks[task_id] = task_name
        self.update_layout()

    async def on_task_complete(self, task_id: str, summary: str) -> None:
        task_name = self.active_tasks.pop(task_id, f"Task {task_id}")
        self.completed_tasks.append(f"{task_name}: {summary}")
        # Keep list reasonably short
        if len(self.completed_tasks) > 20:
            self.completed_tasks.pop(0)
        self.update_layout()

    async def on_vulnerability_found(self, vuln: Vulnerability) -> None:
        severity = vuln.severity.value
        if severity in self.vuln_counts:
            self.vuln_counts[severity] += 1
            
        color = "red" if severity in ["critical", "high"] else "yellow"
        self.reasoning_log.append(f"[{color}][VULN][/] Found {severity.upper()} {vuln.vuln_type.value} at {vuln.endpoint}")
        
        if self.db and self.scan_id:
            self.db.save_vulnerability(self.scan_id, vuln)
            
        self.update_layout()

    async def on_endpoint_discovered(self, method: str, path: str) -> None:
        if self.db and self.scan_id:
            from models.target import Endpoint
            # Construct minimal endpoint for db
            ep = Endpoint(method=method, path=path)
            self.db.save_endpoint(self.scan_id, ep)
        self.total_endpoints += 1
        self.update_layout()

    async def on_llm_reasoning(self, role: str, content: str) -> None:
        """Stream chunks from LLM reasoning."""
        # For simplicity, split into lines and append. 
        # In a real impl, we might buffer chunks into lines.
        clean = content.strip()
        if clean:
            self.reasoning_log.append(f"[{role}] {clean}")
            if len(self.reasoning_log) > 50:
                self.reasoning_log.pop(0)
        self.update_layout()

    async def on_error(self, error: Exception) -> None:
        self.reasoning_log.append(f"[red][ERROR][/red] {str(error)}")
        self.update_layout()
