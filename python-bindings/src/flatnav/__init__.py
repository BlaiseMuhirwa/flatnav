import sys 
from ._core import (
    MetricType,
    utils,
    data_type,
    __version__,
    __doc__
)

class _DataTypeModule:
    from ._core.data_type import DataType

class _UtilsModule:
    from ._core.utils import PruningHeuristic


class _IndexModule:
    from ._core.index import (
        IndexL2Float,
        IndexIPFloat,
        IndexL2Uint8,
        IndexIPUint8,
        IndexL2Int8,
        IndexIPInt8,
        create,
    )


index = _IndexModule
sys.modules['flatnav.index'] = _IndexModule
sys.modules['flatnav.data_type'] = _DataTypeModule
sys.modules['flatnav.utils'] = _UtilsModule

__all__ = [
    'MetricType',
    'data_type',
    'index',
    'utils',
    '__version__',
    '__doc__'
]