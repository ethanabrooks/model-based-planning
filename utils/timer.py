import time
from contextlib import contextmanager
from datetime import timedelta
from typing import Optional

from rich.progress import Progress, ProgressColumn, Task, TextColumn
from rich.text import Text


class TimeElapsedColumn(ProgressColumn):
    """Renders time elapsed."""

    def render(self, task: Task) -> Text:
        """Show time elapsed."""
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("-:--:--", style="progress.elapsed")
        delta = timedelta(seconds=elapsed)
        return Text(str(delta), style="progress.elapsed")


@contextmanager
def Timer(desc: Optional[str] = None):
    columns = [TimeElapsedColumn()]
    if desc:
        columns.append(TextColumn(f"[progress.description]{desc}:"))
    with Progress(*columns) as progress:
        progress.add_task(desc, total=None)
        yield


if __name__ == "__main__":

    def long_running_operation():
        time.sleep(1)  # replace with your operation

    with Timer(desc="sleeping") as timer:
        long_running_operation()
    with Timer() as timer:
        long_running_operation()
