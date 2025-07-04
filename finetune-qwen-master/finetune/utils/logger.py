import io
import sys
import os
import threading
from pathlib import Path
from loguru import logger
import atexit

# 备份原始流
ORIG_STDOUT = sys.stdout
ORIG_STDERR = sys.stderr

try:
    import torch.distributed as dist

    _HAS_DIST = True
except ImportError:
    _HAS_DIST = False


class _StreamLogger(io.TextIOBase):
    def __init__(self, level):
        self.level = level
        self._buffer = ""

    @property
    def encoding(self):
        return ORIG_STDOUT.encoding

    def _emit(self, msg: str):
        # 调用 logger.log 时，临时切回原始 stdout/stderr，避免递归
        orig_out, orig_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = ORIG_STDOUT, ORIG_STDERR
        try:
            logger.log(self.level, msg)
        finally:
            sys.stdout, sys.stderr = orig_out, orig_err

    def write(self, message):
        self._buffer += message
        lines = self._buffer.splitlines(keepends=True)
        self._buffer = ""
        total = 0
        for line in lines:
            total += len(line)
            if line.endswith(("\n", "\r")):
                self._emit(line.rstrip())
            else:
                # 不完整行缓存
                self._buffer = line
        return total

    def flush(self):
        if self._buffer:
            self._emit(self._buffer.rstrip())
            self._buffer = ""


class LoggerConfig:
    """日志配置，只在 rank 0（主进程）打印"""

    _configured = False
    _lock = threading.Lock()
    TIME_STAMP = ""

    @classmethod
    def set_time_stamp(cls, time_stamp: str):
        cls.TIME_STAMP = time_stamp

    @classmethod
    def configure(cls, log_dir: Path, time_stamp: str, logger_name: str = "training"):
        log_dir = Path(log_dir)
        with cls._lock:
            if cls._configured:
                return
            cls._configured = True

            # stamp & dir
            cls.set_time_stamp(time_stamp)
            log_dir.mkdir(parents=True, exist_ok=True)

            # 获取 rank
            if _HAS_DIST and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = int(os.environ.get("RANK", 0))

            # 移除默认 sink
            logger.remove()

            # 主进程才写文件 & 打印 stderr
            if rank == 0:
                fmt = "[{time:YYYY-MM-DD HH:mm:ss}]-[{level}]: {message}"
                log_file = log_dir / f"{logger_name}.log"
                logger.add(sys.__stdout__, level="INFO", format=fmt, enqueue=True)  # 主进程打印到 stdout
                logger.add(log_file.as_posix(), level="DEBUG", format=fmt, enqueue=True)
                logger.add(ORIG_STDERR, level="WARNING", format=fmt, enqueue=True)  # 警告也加入日志

            # 重定向 print 到 logger
            sys.stdout = _StreamLogger("INFO")

            # 绑定 rank，方便扩展
            bound_logger = logger.bind(rank=rank)
            globals()["logger"] = bound_logger
            bound_logger.info(f"Logging initialized on rank {rank}.")

        def _shutdown():
            sys.stdout.flush()
            sys.stderr.flush()
            logger.remove()

        atexit.register(_shutdown)
