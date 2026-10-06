import sys


def pytest_terminal_summary(terminalreporter):
    """Print the final RMSE values at the end of the run, even without -s."""
    mod = sys.modules.get("test_prc") or sys.modules.get("tests.test_prc")
    summary = mod.rmse_summary() if mod else ""
    if summary:
        terminalreporter.write_line("\n" + summary)
