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
    config = ScanConfig(
        target_url=url,
        mode=mode,
        max_duration_minutes=max_duration
    )
    
    dashboard = LiveDashboard(target_url=url)
    orchestrator = Orchestrator(config=config, callbacks=dashboard)
    
    dashboard.start()
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        dashboard.stop()
        console.print("\n[yellow]Scan interrupted by user.[/yellow]")
        sys.exit(130)
    except Exception as e:
        dashboard.stop()
        console.print(f"\n[red]Fatal error during scan: {e}[/red]")
        sys.exit(1)
    finally:
        dashboard.stop()
        
    console.print("\n[green]Scan completed successfully.[/green]")

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
    console.print(f"[yellow]Report generation for {scan_id} is not yet implemented.[/yellow]")

if __name__ == "__main__":
    app()
