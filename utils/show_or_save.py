"""
Utility helpers for handling matplotlib output across different environments.

This module provides a small wrapper to either display plots using an
interactive backend or save them to disk when running in a non-interactive
environment (e.g. headless systems or limited Python distributions).
"""

import matplotlib.pyplot as plt


def show_or_save(filename: str = "output.png") -> None:
    """Display the current matplotlib figure if possible, otherwise save it."""
    backend = plt.get_backend().lower()

    non_interactive_backends = {"agg", "pdf", "svg", "ps", "cairo"}

    if backend in non_interactive_backends:
        plt.savefig(filename)
        print(f"Non-interactive backend detected ({backend}).")
        print(f"Plot saved to '{filename}'.")
    else:
        plt.show()
