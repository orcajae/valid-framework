"""VALID report generation utilities."""
from .checklist import VALIDChecker, VALIDReport


def generate_report(strategy_results, output_path=None):
    """Generate a VALID assessment report from strategy results dict."""
    checker = VALIDChecker()
    report = checker.run_all(**strategy_results)
    if output_path:
        report.to_markdown(output_path)
    return report
