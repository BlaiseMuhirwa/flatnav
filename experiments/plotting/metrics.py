from typing import List, Optional
import numpy as np
from pydantic import BaseModel, validator
import inspect


class MetricConfig(BaseModel):
    description: str
    worst_value: float
    range: Optional[List[float]] = None

    @validator("range")
    def check_range(cls, v):
        if v is not None and (len(v) != 2 or v[0] >= v[1]):
            raise ValueError(
                "range must be a list of two values, where the first is less than the second"
            )
        return v

    class Config:
        extra = "forbid"


class MetricManager:
    def __init__(self):
        self.metric_configs = {}
        self.metric_functions = {}

    def register_metric(
        self, name: str, config: MetricConfig, function: Optional[callable] = None
    ) -> None:
        self.metric_configs[name] = config
        if function:
            self.metric_functions[name] = function

    def get_metric(self, name: str) -> MetricConfig:
        try:
            return self.metric_configs[name]
        except KeyError:
            raise ValueError(f"Metric {name} not found")

    def compute_metric(self, name: str, **kwargs):
        if name not in self.metric_functions:
            raise ValueError(f"Function for metric {name} not found")

        eval_function = self.metric_functions[name]
        sig = inspect.signature(eval_function)
        function_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        return eval_function(**function_kwargs)


def compute_recall(
    queries: np.ndarray, ground_truth: np.ndarray, top_k_indices: List[int], k: int
) -> float:
    ground_truth_sets = [set(gt) for gt in ground_truth]
    mean_recall = 0

    for idx, k_neighbors in enumerate(top_k_indices):
        query_recall = sum(
            1 for neighbor in k_neighbors if neighbor in ground_truth_sets[idx]
        )
        mean_recall += query_recall / k

    recall = mean_recall / len(queries)
    return recall


metric_manager = MetricManager()
metric_manager.register_metric(
    name="recall",
    config=MetricConfig(description="Recall", worst_value=float("-inf"), range=[0, 1]),
    function=compute_recall,
)
metric_manager.register_metric(
    name="qps",
    config=MetricConfig(description="Queries per second", worst_value=float("-inf")),
    function=lambda querying_time, num_queries: num_queries / querying_time,
)
metric_manager.register_metric(
    name="latency_p50",
    config=MetricConfig(
        description="50th percentile latency (ms)", worst_value=float("inf")
    ),
    function=lambda latencies: np.percentile(latencies, 50) * 1000,
)
metric_manager.register_metric(
    name="latency_p90",
    config=MetricConfig(
        description="90th percentile latency (ms)", worst_value=float("inf")
    ),
    function=lambda latencies: np.percentile(latencies, 90) * 1000,
)
metric_manager.register_metric(
    name="latency_p95",
    config=MetricConfig(
        description="95th percentile latency (ms)", worst_value=float("inf")
    ),
    function=lambda latencies: np.percentile(latencies, 95) * 1000,
)
metric_manager.register_metric(
    name="latency_p99",
    config=MetricConfig(
        description="99th percentile latency (ms)", worst_value=float("inf")
    ),
    function=lambda latencies: np.percentile(latencies, 99) * 1000,
)
metric_manager.register_metric(
    name="latency_p999",
    config=MetricConfig(
        description="99.9th percentile latency (ms)", worst_value=float("inf")
    ),
    function=lambda latencies: np.percentile(latencies, 99.9) * 1000,
)
metric_manager.register_metric(
    name="distance_computations",
    config=MetricConfig(
        description="Average number of distance computations per query",
        worst_value=float("inf"),
    ),
    function=lambda distance_computations, num_queries: distance_computations
    / num_queries,
)
metric_manager.register_metric(
    name="index_size",
    config=MetricConfig(description="Index size (bytes)", worst_value=float("inf")),
)
metric_manager.register_metric(
    name="build_time",
    config=MetricConfig(description="Index build time (s)", worst_value=float("inf")),
    function=lambda build_time: build_time,
)
