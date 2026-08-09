"""
Automated report compiler for UAPOML.
Reads verified experimental results and generates Markdown tables and formatted summaries.
"""

from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger("UAPOML_ReportGen")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def generate_markdown_report_tables() -> None:
    tables_dir = Path("reports/tables")
    if not tables_dir.exists():
        logger.warning(f"Tables directory {tables_dir} does not exist yet. Run run_pipeline.py first.")
        return

    csv_files = list(tables_dir.glob("*.csv"))
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            md_path = csv_path.with_suffix(".md")
            df.to_markdown(md_path, index=False)
            logger.info(f"Generated Markdown table: {md_path}")
        except Exception as e:
            logger.error(f"Error converting {csv_path} to markdown: {e}")


if __name__ == "__main__":
    generate_markdown_report_tables()
