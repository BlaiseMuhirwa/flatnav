"""
FlatNav C API Python Bindings

This module provides Python bindings for the FlatNav C API using ctypes.
It allows building and querying nearest neighbor indices from Python.

Example usage:
    import flatnav_c
    import numpy as np

    # Create index
    index = flatnav_c.Index(
        dimension=128,
        max_nodes=100000,
        max_edges=32,
        metric="l2",
        dtype="float32"
    )

    # Add vectors
    vectors = np.random.randn(10000, 128).astype(np.float32)
    index.add_batch(vectors, ef_construction=100, num_init=100)

    # Search
    queries = np.random.randn(100, 128).astype(np.float32)
    labels, distances = index.search(queries, k=10, ef_search=100, num_init=100)
"""

import ctypes
import ctypes.util
import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union

# Type definitions matching C API
class _fn_index_t(ctypes.Structure):
    """Opaque index handle"""
    pass

class _fn_distance_t(ctypes.Structure):
    """Opaque distance handle"""
    pass

class _fn_result_t(ctypes.Structure):
    """Search result (distance, label pair)"""
    _fields_ = [
        ("distance", ctypes.c_float),
        ("label", ctypes.c_int32),
    ]

class _fn_index_config_t(ctypes.Structure):
    """Index configuration"""
    _fields_ = [
        ("dimension", ctypes.c_uint32),
        ("max_node_count", ctypes.c_uint32),
        ("max_edges_per_node", ctypes.c_uint32),
        ("metric", ctypes.c_int),
        ("data_type", ctypes.c_int),
        ("collect_stats", ctypes.c_uint8),
        ("_reserved", ctypes.c_uint8 * 3),
    ]

class _fn_search_params_t(ctypes.Structure):
    """Search parameters"""
    _fields_ = [
        ("k", ctypes.c_uint32),
        ("ef_search", ctypes.c_uint32),
        ("num_initializations", ctypes.c_uint32),
    ]

# Error codes
FN_ERR_OK = 0
FN_ERR_NULL_PTR = -1
FN_ERR_INVALID_ARG = -2
FN_ERR_OUT_OF_MEMORY = -3
FN_ERR_INDEX_FULL = -4
FN_ERR_IO_ERROR = -5
FN_ERR_INVALID_FORMAT = -6
FN_ERR_DIMENSION_MISMATCH = -7
FN_ERR_NOT_SUPPORTED = -8
FN_ERR_NOT_INITIALIZED = -9
FN_ERR_INTERNAL = -99

# Data types
FN_DATA_FLOAT32 = 0
FN_DATA_INT8 = 1
FN_DATA_UINT8 = 2

# Metric types
FN_METRIC_L2 = 0
FN_METRIC_IP = 1

# Map string names to enum values
DTYPE_MAP = {
    "float32": FN_DATA_FLOAT32,
    "float": FN_DATA_FLOAT32,
    "int8": FN_DATA_INT8,
    "uint8": FN_DATA_UINT8,
}

METRIC_MAP = {
    "l2": FN_METRIC_L2,
    "euclidean": FN_METRIC_L2,
    "ip": FN_METRIC_IP,
    "inner_product": FN_METRIC_IP,
    "angular": FN_METRIC_IP,
}

NUMPY_DTYPE_MAP = {
    FN_DATA_FLOAT32: np.float32,
    FN_DATA_INT8: np.int8,
    FN_DATA_UINT8: np.uint8,
}


def _find_library() -> str:
    """Find the flatnav_c shared library."""
    # Possible library names
    if sys.platform == "darwin":
        lib_names = ["libflatnav_c.dylib", "libflatnav_c.so"]
    elif sys.platform == "win32":
        lib_names = ["flatnav_c.dll", "libflatnav_c.dll"]
    else:
        lib_names = ["libflatnav_c.so"]

    # Search paths (in order of preference)
    search_paths = []

    # 1. Package directory
    pkg_dir = Path(__file__).parent
    search_paths.append(pkg_dir)
    search_paths.append(pkg_dir / "lib")

    # 2. Build directory (for development)
    build_dir = pkg_dir.parent.parent / "build"
    search_paths.append(build_dir)

    # 3. Standard library paths
    for path in ["/usr/local/lib", "/usr/lib"]:
        search_paths.append(Path(path))

    # 4. LD_LIBRARY_PATH
    if "LD_LIBRARY_PATH" in os.environ:
        for path in os.environ["LD_LIBRARY_PATH"].split(":"):
            search_paths.append(Path(path))

    # Search for library
    for search_path in search_paths:
        for lib_name in lib_names:
            lib_path = search_path / lib_name
            if lib_path.exists():
                return str(lib_path)

    # Try ctypes.util.find_library as last resort
    lib_path = ctypes.util.find_library("flatnav_c")
    if lib_path:
        return lib_path

    raise RuntimeError(
        "Could not find libflatnav_c shared library. "
        "Make sure to build with -DFN_BUILD_SHARED=ON and install or "
        "set LD_LIBRARY_PATH to include the build directory."
    )


def _load_library() -> ctypes.CDLL:
    """Load the FlatNav C library."""
    lib_path = _find_library()
    lib = ctypes.CDLL(lib_path)

    # Define function signatures
    # Library initialization
    lib.fn_init.restype = None
    lib.fn_init.argtypes = []

    lib.fn_cleanup.restype = None
    lib.fn_cleanup.argtypes = []

    lib.fn_is_initialized.restype = ctypes.c_int
    lib.fn_is_initialized.argtypes = []

    # Version info
    lib.fn_version_string.restype = ctypes.c_char_p
    lib.fn_version_string.argtypes = []

    lib.fn_build_info.restype = ctypes.c_char_p
    lib.fn_build_info.argtypes = []

    # Error string
    lib.fn_error_string.restype = ctypes.c_char_p
    lib.fn_error_string.argtypes = [ctypes.c_int]

    # Index lifecycle
    lib.fn_index_create.restype = ctypes.c_int
    lib.fn_index_create.argtypes = [
        ctypes.POINTER(ctypes.POINTER(_fn_index_t)),
        ctypes.POINTER(_fn_index_config_t)
    ]

    lib.fn_index_destroy.restype = None
    lib.fn_index_destroy.argtypes = [ctypes.POINTER(_fn_index_t)]

    # Index operations
    lib.fn_index_add.restype = ctypes.c_int
    lib.fn_index_add.argtypes = [
        ctypes.POINTER(_fn_index_t),
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_uint32,
        ctypes.c_uint32
    ]

    lib.fn_index_search.restype = ctypes.c_int
    lib.fn_index_search.argtypes = [
        ctypes.POINTER(_fn_index_t),
        ctypes.c_void_p,
        ctypes.POINTER(_fn_result_t),
        ctypes.POINTER(_fn_search_params_t)
    ]

    # Index properties
    lib.fn_index_get_dimension.restype = ctypes.c_uint32
    lib.fn_index_get_dimension.argtypes = [ctypes.POINTER(_fn_index_t)]

    lib.fn_index_get_num_nodes.restype = ctypes.c_uint32
    lib.fn_index_get_num_nodes.argtypes = [ctypes.POINTER(_fn_index_t)]

    lib.fn_index_get_max_nodes.restype = ctypes.c_uint32
    lib.fn_index_get_max_nodes.argtypes = [ctypes.POINTER(_fn_index_t)]

    # Serialization
    lib.fn_index_save.restype = ctypes.c_int
    lib.fn_index_save.argtypes = [ctypes.POINTER(_fn_index_t), ctypes.c_char_p]

    lib.fn_index_load.restype = ctypes.c_int
    lib.fn_index_load.argtypes = [
        ctypes.POINTER(ctypes.POINTER(_fn_index_t)),
        ctypes.c_char_p
    ]

    return lib


# Global library instance
_lib = None


def _get_lib() -> ctypes.CDLL:
    """Get or load the library."""
    global _lib
    if _lib is None:
        _lib = _load_library()
        _lib.fn_init()
    return _lib


class FlatNavError(Exception):
    """Exception raised by FlatNav operations."""
    def __init__(self, error_code: int, message: str = ""):
        self.error_code = error_code
        lib = _get_lib()
        error_str = lib.fn_error_string(error_code).decode("utf-8")
        super().__init__(f"{error_str}" + (f": {message}" if message else ""))


def _check_error(err: int, context: str = ""):
    """Check error code and raise exception if not OK."""
    if err != FN_ERR_OK:
        raise FlatNavError(err, context)


def version() -> str:
    """Get FlatNav C API version string."""
    lib = _get_lib()
    return lib.fn_version_string().decode("utf-8")


def build_info() -> str:
    """Get build configuration info."""
    lib = _get_lib()
    return lib.fn_build_info().decode("utf-8")


class Index:
    """
    FlatNav nearest neighbor search index.

    Parameters
    ----------
    dimension : int
        Vector dimension.
    max_nodes : int
        Maximum number of vectors the index can hold.
    max_edges : int
        Maximum edges per node (M parameter). Default: 32.
    metric : str
        Distance metric: "l2" (default) or "ip" (inner product).
    dtype : str
        Data type: "float32" (default), "int8", or "uint8".
    collect_stats : bool
        Whether to collect search statistics. Default: False.

    Examples
    --------
    >>> index = Index(dimension=128, max_nodes=10000, max_edges=32)
    >>> vectors = np.random.randn(1000, 128).astype(np.float32)
    >>> for i, vec in enumerate(vectors):
    ...     index.add(vec, label=i)
    >>> labels, distances = index.search(vectors[:10], k=10)
    """

    def __init__(
        self,
        dimension: int,
        max_nodes: int,
        max_edges: int = 32,
        metric: str = "l2",
        dtype: str = "float32",
        collect_stats: bool = False,
    ):
        self._lib = _get_lib()
        self._index = None
        self._dimension = dimension
        self._dtype = DTYPE_MAP.get(dtype.lower())
        self._metric = METRIC_MAP.get(metric.lower())

        if self._dtype is None:
            raise ValueError(f"Unknown dtype: {dtype}. Use: {list(DTYPE_MAP.keys())}")
        if self._metric is None:
            raise ValueError(f"Unknown metric: {metric}. Use: {list(METRIC_MAP.keys())}")

        # Create config
        config = _fn_index_config_t()
        config.dimension = dimension
        config.max_node_count = max_nodes
        config.max_edges_per_node = max_edges
        config.metric = self._metric
        config.data_type = self._dtype
        config.collect_stats = 1 if collect_stats else 0

        # Create index
        index_ptr = ctypes.POINTER(_fn_index_t)()
        err = self._lib.fn_index_create(ctypes.byref(index_ptr), ctypes.byref(config))
        _check_error(err, "Failed to create index")

        self._index = index_ptr

    def __del__(self):
        """Destroy the index."""
        if self._index is not None and self._lib is not None:
            self._lib.fn_index_destroy(self._index)
            self._index = None

    @property
    def dimension(self) -> int:
        """Vector dimension."""
        return self._lib.fn_index_get_dimension(self._index)

    @property
    def num_vectors(self) -> int:
        """Number of vectors in the index."""
        return self._lib.fn_index_get_num_nodes(self._index)

    @property
    def max_vectors(self) -> int:
        """Maximum capacity of the index."""
        return self._lib.fn_index_get_max_nodes(self._index)

    def _ensure_array(
        self, data: np.ndarray, expected_shape: Optional[Tuple[int, ...]] = None
    ) -> np.ndarray:
        """Ensure array is contiguous and correct dtype."""
        target_dtype = NUMPY_DTYPE_MAP[self._dtype]
        if data.dtype != target_dtype:
            data = data.astype(target_dtype)
        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)
        if expected_shape and data.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {data.shape}")
        return data

    def add(
        self,
        vector: np.ndarray,
        label: int,
        ef_construction: int = 100,
        num_init: int = 100,
    ) -> None:
        """
        Add a single vector to the index.

        Parameters
        ----------
        vector : np.ndarray
            Vector to add (shape: [dimension]).
        label : int
            Label/ID for the vector.
        ef_construction : int
            Construction beam width. Default: 100.
        num_init : int
            Number of random starting points. Default: 100.
        """
        vector = self._ensure_array(vector, (self._dimension,))
        err = self._lib.fn_index_add(
            self._index,
            vector.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int32(label),
            ctypes.c_uint32(ef_construction),
            ctypes.c_uint32(num_init),
        )
        _check_error(err, f"Failed to add vector with label {label}")

    def add_batch(
        self,
        vectors: np.ndarray,
        labels: Optional[np.ndarray] = None,
        ef_construction: int = 100,
        num_init: int = 100,
        num_threads: int = 1,
    ) -> None:
        """
        Add multiple vectors to the index.

        Parameters
        ----------
        vectors : np.ndarray
            Vectors to add (shape: [n, dimension]).
        labels : np.ndarray, optional
            Labels for vectors. If None, uses 0, 1, 2, ...
        ef_construction : int
            Construction beam width. Default: 100.
        num_init : int
            Number of random starting points. Default: 100.
        num_threads : int
            Number of threads for parallel indexing. Default: 1.
        """
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got {vectors.ndim}D")
        if vectors.shape[1] != self._dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self._dimension}, got {vectors.shape[1]}"
            )

        n = vectors.shape[0]
        vectors = self._ensure_array(vectors)

        if labels is None:
            start_label = self.num_vectors
            labels = np.arange(start_label, start_label + n, dtype=np.int32)
        else:
            labels = np.asarray(labels, dtype=np.int32)
            if len(labels) != n:
                raise ValueError(f"Labels length {len(labels)} != vectors count {n}")

        # Add vectors one by one (thread-safe via C API)
        for i in range(n):
            self.add(vectors[i], int(labels[i]), ef_construction, num_init)

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        ef_search: int = 100,
        num_init: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.

        Parameters
        ----------
        queries : np.ndarray
            Query vectors (shape: [dimension] or [n, dimension]).
        k : int
            Number of neighbors to return. Default: 10.
        ef_search : int
            Search beam width. Default: 100.
        num_init : int
            Number of random starting points. Default: 100.

        Returns
        -------
        labels : np.ndarray
            Labels of nearest neighbors (shape: [n, k] or [k]).
        distances : np.ndarray
            Distances to nearest neighbors (shape: [n, k] or [k]).
        """
        single_query = queries.ndim == 1
        if single_query:
            queries = queries.reshape(1, -1)

        if queries.shape[1] != self._dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self._dimension}, got {queries.shape[1]}"
            )

        queries = self._ensure_array(queries)
        n_queries = queries.shape[0]

        # Create search params
        params = _fn_search_params_t()
        params.k = k
        params.ef_search = ef_search
        params.num_initializations = num_init

        # Allocate results
        labels = np.zeros((n_queries, k), dtype=np.int32)
        distances = np.zeros((n_queries, k), dtype=np.float32)

        # Search each query
        results = (_fn_result_t * k)()
        for i in range(n_queries):
            query_ptr = queries[i : i + 1].ctypes.data_as(ctypes.c_void_p)
            err = self._lib.fn_index_search(
                self._index, query_ptr, results, ctypes.byref(params)
            )
            _check_error(err, f"Search failed for query {i}")

            for j in range(k):
                labels[i, j] = results[j].label
                distances[i, j] = results[j].distance

        if single_query:
            return labels[0], distances[0]
        return labels, distances

    def save(self, path: str) -> None:
        """
        Save index to file.

        Parameters
        ----------
        path : str
            Path to save the index.
        """
        err = self._lib.fn_index_save(self._index, path.encode("utf-8"))
        _check_error(err, f"Failed to save index to {path}")

    @classmethod
    def load(cls, path: str) -> "Index":
        """
        Load index from file.

        Parameters
        ----------
        path : str
            Path to load the index from.

        Returns
        -------
        Index
            Loaded index.
        """
        lib = _get_lib()
        index_ptr = ctypes.POINTER(_fn_index_t)()
        err = lib.fn_index_load(ctypes.byref(index_ptr), path.encode("utf-8"))
        _check_error(err, f"Failed to load index from {path}")

        # Create wrapper object
        obj = cls.__new__(cls)
        obj._lib = lib
        obj._index = index_ptr
        obj._dimension = lib.fn_index_get_dimension(index_ptr)
        obj._dtype = FN_DATA_FLOAT32  # Default, should be read from index
        obj._metric = FN_METRIC_L2  # Default, should be read from index
        return obj

    def __repr__(self) -> str:
        return (
            f"Index(dimension={self.dimension}, "
            f"num_vectors={self.num_vectors}, "
            f"max_vectors={self.max_vectors})"
        )
