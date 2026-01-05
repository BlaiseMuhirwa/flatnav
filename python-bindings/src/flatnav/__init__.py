import sys
from ._core import (
    MetricType,
    TwoPassStrategy,
    Pass2CandidateMethod,
    data_type,
    __version__,
    __doc__
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
    )


index = _IndexModule
sys.modules['flatnav.index'] = _IndexModule
sys.modules['flatnav.data_type'] = _DataTypeModule

__all__ = [
    'MetricType',
    'TwoPassStrategy',
    'Pass2CandidateMethod',
    'data_type',
    'index',
    '__version__',
    '__doc__'
]