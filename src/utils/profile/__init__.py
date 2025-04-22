import time
import contextlib
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class TimeSpan:
    name: str
    start: float
    end: float = field(default=-1)
    children: List["TimeSpan"] = field(default_factory=list)
    parent: Optional["TimeSpan"] = None

    @property
    def duration_ms(self) -> float:
        if self.end == -1:
            return -1
        return (self.end - self.start) * 1000

    def __str__(self) -> str:
        return f"{self.name}: {self.duration_ms:.2f}ms"


class Profiler:
    def __init__(self):
        self._spans: Dict[int, List[TimeSpan]] = defaultdict(list)
        self._active_spans: Dict[int, Optional[TimeSpan]] = {}
        self._lock = threading.Lock()

    def start_span(self, name: str) -> TimeSpan:
        thread_id = threading.get_ident()
        span = TimeSpan(name, time.perf_counter())

        with self._lock:
            active = self._active_spans.get(thread_id)
            if active:
                span.parent = active
                active.children.append(span)
            else:
                self._spans[thread_id].append(span)
            self._active_spans[thread_id] = span

        return span

    def end_span(self, span: TimeSpan):
        span.end = time.perf_counter()
        thread_id = threading.get_ident()

        with self._lock:
            if span.parent:
                self._active_spans[thread_id] = span.parent
            else:
                self._active_spans.pop(thread_id, None)

    def get_spans(self) -> List[TimeSpan]:
        """Get current spans for this thread"""
        thread_id = threading.get_ident()
        with self._lock:
            return self._spans.get(thread_id, []).copy()

    def clear(self):
        """Clear profiling data for this thread"""
        thread_id = threading.get_ident()
        with self._lock:
            self._spans[thread_id] = []
            self._active_spans[thread_id] = None


# Global instance
_profiler = Profiler()


@contextlib.contextmanager
def profile(name: str):
    """Simple context manager for profiling"""
    span = _profiler.start_span(name)
    try:
        yield span
    finally:
        _profiler.end_span(span)


def get_spans() -> List[TimeSpan]:
    """Get current profiling spans"""
    return _profiler.get_spans()


def format_spans(spans: List[TimeSpan], indent: int = 0) -> None:
    """Format and print spans in a tree structure"""
    for span in spans:
        print("  " * indent + str(span))
        if span.children:
            format_spans(span.children, indent + 1)


def print_spans():
    format_spans(get_spans())


def clear_profile():
    """Clear profiling data"""
    _profiler.clear()
