import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ._core import (
    MetricType,
    TwoPassStrategy,
    Pass2CandidateMethod,
    data_type,
    __version__,
    __doc__
)
from ._core.perf import reset_perf_counters, get_perf_counters
from ._core.data_type import DataType


@dataclass
class TwoPassConfig:
    """Configuration for two-pass index construction."""
    strategy: TwoPassStrategy = TwoPassStrategy.HUBNESS_SCORING
    M_pass1: int = 4
    M_pass2: int = 4
    ef_construction_pass1: int = 10
    ef_construction_pass2: int = 100
    hubness_penalty_weight: float = 0.1
    edge_quality_threshold: float = 0.5
    pass2_candidate_method: Pass2CandidateMethod = Pass2CandidateMethod.BEAM_SEARCH
    neighbor_expansion_hops: int = 2


@dataclass
class AnchorConfig:
    """Configuration for anchor-based index construction."""
    anchor_fraction: float = 0.01
    M: int = 16
    anchor_ef_construction: int = 500
    bulk_ef_construction: int = 80
    num_anchor_probes: int = 10
    seed: int = 42
    quantization: Optional["SQConfig"] = None


@dataclass
class SQConfig:
    """Configuration for scalar-quantized index construction."""
    training_data: np.ndarray = field(repr=False)
    max_edges_per_node: int = 32
    max_train_samples: int = 0
    ef_construction: int = 100


def build(
    distance_type: str,
    dim: int,
    dataset_size: int,
    data: np.ndarray,
    config=None,
    max_edges_per_node: int = 32,
    ef_construction: int = 100,
    num_initializations: int = 100,
    num_threads: int = 1,
    labels=None,
    verbose: bool = False,
    collect_stats: bool = False,
    index_data_type: DataType = DataType.float32,
):
    """Build a flatnav index.

    Parameters
    ----------
    distance_type : str
        Distance metric: 'l2' or 'angular'
    dim : int
        Dimension of the vectors.
    dataset_size : int
        Number of vectors to index.
    data : numpy.ndarray
        2D array of vectors to index (shape: [dataset_size, dim]).
    config : TwoPassConfig | AnchorConfig | SQConfig | None
        Construction method config. None uses standard single-pass construction.
    max_edges_per_node : int
        Maximum edges per node (standard construction only, default: 32).
    ef_construction : int
        ef_construction (standard construction only, default: 100).
    num_initializations : int
        Number of random entry points for search (default: 100).
    num_threads : int
        Number of threads for parallel construction (default: 1).
    labels : array-like, optional
        Labels for each vector (default: auto-generated 0 to N-1).
    verbose : bool
        Enable verbose output (default: False).
    collect_stats : bool
        Collect construction statistics (default: False).
    index_data_type : DataType
        Data type for index storage (default: DataType.float32).

    Returns
    -------
    Index
        The constructed index.
    """
    from ._core import index as _index

    if config is None:
        # Standard single-pass construction
        idx = _index.create(
            distance_type=distance_type,
            dim=dim,
            dataset_size=dataset_size,
            max_edges_per_node=max_edges_per_node,
            index_data_type=index_data_type,
            verbose=verbose,
            collect_stats=collect_stats,
        )
        idx.set_num_threads(num_threads)
        idx.add(data, ef_construction, num_initializations, labels)
        return idx

    if isinstance(config, TwoPassConfig):
        return _index.create_two_pass(
            distance_type=distance_type,
            dim=dim,
            dataset_size=dataset_size,
            data=data,
            strategy=config.strategy,
            M_pass1=config.M_pass1,
            M_pass2=config.M_pass2,
            ef_construction_pass1=config.ef_construction_pass1,
            ef_construction_pass2=config.ef_construction_pass2,
            num_initializations=num_initializations,
            hubness_penalty_weight=config.hubness_penalty_weight,
            edge_quality_threshold=config.edge_quality_threshold,
            num_threads=num_threads,
            pass2_candidate_method=config.pass2_candidate_method,
            neighbor_expansion_hops=config.neighbor_expansion_hops,
            index_data_type=index_data_type,
            labels=labels,
        )

    if isinstance(config, AnchorConfig):
        if config.quantization is not None:
            sq = config.quantization
            return _index.create_anchor_sq(
                distance_type=distance_type,
                dim=dim,
                dataset_size=dataset_size,
                training_data=sq.training_data,
                max_train_samples=sq.max_train_samples,
                data=data,
                anchor_fraction=config.anchor_fraction,
                M=config.M,
                anchor_ef_construction=config.anchor_ef_construction,
                bulk_ef_construction=config.bulk_ef_construction,
                num_anchor_probes=config.num_anchor_probes,
                num_initializations=num_initializations,
                num_threads=num_threads,
                seed=config.seed,
                collect_stats=collect_stats,
                labels=labels,
            )
        return _index.create_anchor(
            distance_type=distance_type,
            dim=dim,
            dataset_size=dataset_size,
            data=data,
            anchor_fraction=config.anchor_fraction,
            M=config.M,
            anchor_ef_construction=config.anchor_ef_construction,
            bulk_ef_construction=config.bulk_ef_construction,
            num_anchor_probes=config.num_anchor_probes,
            num_initializations=num_initializations,
            num_threads=num_threads,
            seed=config.seed,
            collect_stats=collect_stats,
            index_data_type=index_data_type,
            labels=labels,
        )

    if isinstance(config, SQConfig):
        idx = _index.create_sq(
            distance_type=distance_type,
            dim=dim,
            dataset_size=dataset_size,
            max_edges_per_node=config.max_edges_per_node,
            training_data=config.training_data,
            max_train_samples=config.max_train_samples,
            verbose=verbose,
            collect_stats=collect_stats,
        )
        idx.set_num_threads(num_threads)
        idx.add(data, config.ef_construction, num_initializations, labels)
        return idx

    raise TypeError(
        f"config must be TwoPassConfig, AnchorConfig, SQConfig, or None, "
        f"got {type(config).__name__}"
    )


class _DataTypeModule:
    from ._core.data_type import DataType


class _IndexModule:
    from ._core.index import (
        IndexL2Float,
        IndexIPFloat,
        IndexL2Uint8,
        IndexIPUint8,
        IndexL2Int8,
        IndexIPInt8,
        create,
        create_two_pass,
        create_anchor,
        create_anchor_sq,
        create_sq,
    )


_IndexModule.build = staticmethod(build)

index = _IndexModule
sys.modules['flatnav.index'] = _IndexModule
sys.modules['flatnav.data_type'] = _DataTypeModule

__all__ = [
    'MetricType',
    'TwoPassStrategy',
    'Pass2CandidateMethod',
    'DataType',
    'TwoPassConfig',
    'AnchorConfig',
    'SQConfig',
    'build',
    'data_type',
    'index',
    'reset_perf_counters',
    'get_perf_counters',
    '__version__',
    '__doc__'
]