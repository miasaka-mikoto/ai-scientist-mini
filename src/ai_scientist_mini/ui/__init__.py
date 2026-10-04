"""User interface for :mod:`ai_scientist_mini`.

The UI is deliberately kept separate from the research engine.  The
``DashboardController`` talks to a small, duck-typed service interface, so a
real engine can be plugged in without making Tkinter (or any other desktop
toolkit) part of the scientific core.
"""

from .dashboard import (
    Dashboard,
    DashboardController,
    DashboardSnapshot,
    DashboardViewModel,
    LocalDemoService,
    StudyServiceAdapter,
)

__all__ = [
    "Dashboard",
    "DashboardController",
    "DashboardSnapshot",
    "DashboardViewModel",
    "LocalDemoService",
    "StudyServiceAdapter",
]
