import threading
import time
from typing import Optional


class Timer:
    def __init__(self, desc: Optional[str] = None, print_freq: float = 0.01):
        self.start_time = None
        self.thread = threading.Thread(target=self._print_elapsed_time)
        self.thread.daemon = True
        self.stop_flag = False
        self.desc = desc
        self.print_freq = print_freq

    def __enter__(self):
        self.start_time = time.time()
        self.thread.start()

    def __exit__(self, *_, **__):
        self.stop_flag = True
        self.thread.join()

    def _print_elapsed_time(self):
        while not self.stop_flag:
            elapsed_time = time.time() - self.start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            msg = f"{seconds:.2f}"
            if minutes > 0 or hours > 0:
                msg = f"{int(minutes):02d}:{msg}"
            else:
                msg = f"{msg} s"
            if hours > 0:
                msg = f"{int(hours):02d}:{msg}"
            if self.desc:
                msg = f"{self.desc}: {msg}"
            print(f"\r{msg}", end="")
            time.sleep(self.print_freq)  # adjust the printing frequency as needed
        print()


def long_running_operation():
    time.sleep(1)  # replace with your operation


if __name__ == "__main__":
    with Timer(desc="sleeping") as timer:
        long_running_operation()
    with Timer() as timer:
        long_running_operation()
