"""Lightweight file logger for SENK training and inference scripts.

Writes each ``info`` message both to stdout and to a timestamped log file
inside ``output_dir``.  When called in a distributed setting, pass
``is_rank0=False`` on non-rank0 processes to suppress their output.
"""
import os
import datetime


class FileLogger:
    """Minimal logger that mirrors messages to stdout and a log file.

    Args:
        is_master: When ``False``, all messages are silently dropped.
            Reserved for multi-GPU training where only the master process
            should emit logs.
        is_rank0: When ``False``, messages are silently dropped.
            Reserved for DistributedDataParallel workers other than rank 0.
        output_dir: Directory in which to create the timestamped log file.
            Defaults to the current working directory.
    """

    def __init__(self, is_master: bool = True, is_rank0: bool = True, output_dir: str = None):
        self.is_master = is_master
        self.is_rank0 = is_rank0
        self.output_dir = output_dir or os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)
        time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(self.output_dir, f'log_{time_str}.txt')
        self._fh = None
        if self.is_master and self.is_rank0:
            self._fh = open(self.log_path, 'a', buffering=1)

    def info(self, msg) -> None:
        """Emit an informational message to stdout and the log file.

        Non-master or non-rank0 processes drop the message silently.
        """
        if not (self.is_master and self.is_rank0):
            return
        if not isinstance(msg, str):
            msg = str(msg)
        print(msg)
        if self._fh is not None:
            try:
                self._fh.write(msg + '\n')
            except Exception:
                pass

    def close(self) -> None:
        """Close the underlying log file handle (no-op if already closed)."""
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
