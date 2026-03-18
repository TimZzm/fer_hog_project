from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable


@dataclass
class BenchmarkResult:
    stage: str
    seconds: float
    items: int | None = None

    @property
    def seconds_per_item(self) -> float | None:
        if not self.items:
            return None
        return self.seconds / self.items

    @property
    def throughput(self) -> float | None:
        if not self.items or self.seconds == 0.0:
            return None
        return self.items / self.seconds


def timed_call(func: Callable[..., Any], *args, **kwargs) -> tuple[Any, BenchmarkResult]:
    stage = kwargs.pop("_stage_name", getattr(func, "__name__", "stage"))
    items = kwargs.pop("_items", None)
    start = perf_counter()
    result = func(*args, **kwargs)
    elapsed = perf_counter() - start
    return result, BenchmarkResult(stage=stage, seconds=elapsed, items=items)


def format_benchmark_table(results: list[BenchmarkResult]) -> str:
    header = (
        f"{'Stage':<28} {'Seconds':>12} {'Items':>10} {'Sec/Image':>12} {'Img/Sec':>12}"
    )
    lines = [header, "-" * len(header)]
    for result in results:
        sec_per_item = result.seconds_per_item
        throughput = result.throughput
        lines.append(
            f"{result.stage:<28} "
            f"{result.seconds:>12.4f} "
            f"{'' if result.items is None else result.items:>10} "
            f"{'' if sec_per_item is None else f'{sec_per_item:.6f}':>12} "
            f"{'' if throughput is None else f'{throughput:.2f}':>12}"
        )
    return "\n".join(lines)
