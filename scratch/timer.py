import time
from rich.progress import Progress, TimeElapsedColumn

progress = Progress(
    TimeElapsedColumn(), *Progress.get_default_columns(), auto_refresh=False
)
# master_task = progress.add_task("Sorting data...", total=3)
sorted_data = []
with progress:
    for d in progress.track([1, 2, 3]):
        progress.add_task("Sorting...", total=None)
        time.sleep(3)
        # progress.advance(1)
