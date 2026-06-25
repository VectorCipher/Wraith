import asyncio
import sys
from typing import Optional

import typer
from rich.console import Console

from cli.display import LiveDashboard
from core.orchestrator import Orchestrator
from models.scan import ScanConfig, ScanMode
from llm.model_manager import ModelManager

app = typer.Typer(
    name="wraith",
    help="WRAITH Autonomous AI Penetration Testing Agent",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

@app.command()
def scan(
    url: str = typer.Argument(..., help="Target URL to scan"),
    mode: ScanMode = typer.Option(
        ScanMode.FULL, 
        "--mode", "-m", 
        help="Scan mode (fast, full, whitebox)"
    ),
    max_duration: int = typer.Option(
        60,
        "--max-duration", "-d",
        help="Maximum time budget in minutes"
    ),
):
    """
    Start an autonomous penetration test against a target URL.
    """
    from config import settings
    
    console.print("\n[cyan]WRAITH requires an LLM to act as its reasoning engine.[/cyan]")
    console.print("Please select your LLM provider:")
    console.print("  [1] Ollama (Local, requires 'ollama serve' running)")
    console.print("  [2] OpenRouter (Cloud, requires API key)")
    
    choice = typer.prompt("Enter your choice (1 or 2)", default="1")
    
    if choice == "2":
        settings.llm_provider = "openrouter"
        api_key = typer.prompt("Enter your OpenRouter API Key", hide_input=True)
        if not api_key.strip():
            console.print("[red]API Key is required for OpenRouter.[/red]")
            raise typer.Abort()
        settings.openrouter_api_key = api_key.strip()
        
        model_name = typer.prompt("Enter the OpenRouter model name (e.g. anthropic/claude-3-sonnet)", default="anthropic/claude-3-sonnet")
        if model_name.strip():
            settings.model = model_name.strip()
    else:
        settings.llm_provider = "ollama"
        # Optional: We could do a quick health check here for Ollama,
        # but the client initialization will catch connection errors anyway.

    config = ScanConfig(
        target_url=url,
        mode=mode,
        max_duration_minutes=max_duration
    )
    
    from databases.db import DatabaseManager
    db = DatabaseManager()
    
    dashboard = LiveDashboard(target_url=url, db_manager=db)
    orchestrator = Orchestrator(config=config, callbacks=dashboard)
    dashboard.scan_id = orchestrator.scan_id
    
    # Persist initial scan state
    db.save_scan(scan_id=orchestrator.scan_id, target_url=url, status="running", config=config.model_dump())
    
    dashboard.start()
    try:
        asyncio.run(orchestrator.run())
        db.update_scan_status(orchestrator.scan_id, "completed", complete=True)
    except KeyboardInterrupt:
        dashboard.stop()
        db.update_scan_status(orchestrator.scan_id, "aborted", complete=True)
        console.print("\n[yellow]Scan interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        dashboard.stop()
        db.update_scan_status(orchestrator.scan_id, "failed", complete=True)
        console.print(f"\n[red]Fatal error during scan: {e}[/red]")
        sys.exit(1)
    finally:
        dashboard.stop()
        
    console.print(f"\n[green]Scan completed successfully. Run `wraith report {orchestrator.scan_id}` to view results.[/green]")

@app.command()
def models():
    """
    List available models and check their health.
    """
    console.print("[cyan]Checking Ollama models...[/cyan]")
    from llm.client import LLMClient
    
    async def check_models():
        client = LLMClient()
        manager = ModelManager(client=client)
        
        try:
            status = await manager.get_status_display()
            for model in status:
                color = "green" if model["status"] == "ready" else "red" if model["status"] == "unhealthy" else "yellow"
                console.print(f"- {model['model']}: [{color}]{model['status']}[/{color}] (Role: {model['role']})")
        finally:
            await client.close()
            
    asyncio.run(check_models())

@app.command()
def report(scan_id: str):
    """
    Generate a report from a previous scan.
    """
    from databases.db import DatabaseManager
    from reporters.html_reporter import HtmlReporter
    
    console.print(f"[cyan]Generating report for {scan_id}...[/cyan]")
    try:
        db = DatabaseManager()
        reporter = HtmlReporter(db)
        report_path = reporter.generate_report(scan_id)
        console.print(f"[green]✔ Report generated successfully![/green]")
        console.print(f"Path: {report_path}")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Failed to generate report: {e}[/red]")

if __name__ == "__main__":
    app()
