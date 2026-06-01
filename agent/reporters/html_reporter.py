"""
WRAITH HTML Reporter.

Fetches data from the SQLite database and generates a professional
HTML penetration test report using Jinja2 templates.
"""

import os
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from databases.db import DatabaseManager
from utils.logger import get_logger

logger = get_logger("reporters.html")

class HtmlReporter:
    """Generates HTML reports from scan data."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        
        # Setup Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        # Ensure reports directory exists
        home = Path.home()
        self.reports_dir = home / ".wraith" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, scan_id: str) -> str:
        """
        Generate an HTML report for the given scan_id.
        Returns the absolute path to the generated HTML file.
        """
        scan = self.db.get_scan(scan_id)
        if not scan:
            raise ValueError(f"Scan ID {scan_id} not found in database.")

        endpoints = self.db.get_endpoints(scan_id)
        vulns = self.db.get_vulnerabilities(scan_id)

        # Prepare template context
        context = {
            "scan_id": scan_id,
            "target_url": scan["target_url"],
            "start_time": scan["start_time"],
            "end_time": scan["end_time"],
            "status": scan["status"],
            "endpoints": endpoints,
            "vulnerabilities": vulns,
            "critical_count": sum(1 for v in vulns if v.severity.value == "critical"),
            "high_count": sum(1 for v in vulns if v.severity.value == "high"),
            "medium_count": sum(1 for v in vulns if v.severity.value == "medium"),
            "low_count": sum(1 for v in vulns if v.severity.value == "low"),
            "info_count": sum(1 for v in vulns if v.severity.value == "info"),
        }

        # Render template
        template = self.env.get_template("report.html")
        html_content = template.render(**context)

        # Save to file
        report_path = self.reports_dir / f"wraith_report_{scan_id}.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Generated HTML report at {report_path}")
        return str(report_path)
