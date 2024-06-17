from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Iterator,
    Literal,
    Self,
    Tuple,
    TypeVar,
)

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:

    class Frequency:
        f: NDArray[np.float_]

    class Network:
        name: str
        params: dict[Any, Any]
        frequency: Frequency
        s_mag: NDArray[np.float_]

        def __getitem__(self, *args: Any) -> Self: ...
        def __pow__(self, args: Any) -> Self: ...
        def plot_s_smith(self, **kwargs: Any) -> None: ...
        def plot_s_vswr(self, **kwargs: Any) -> None: ...

    class NetworkSet:
        name: str

        def __init__(self, networks: list[Network], name: str = ""): ...
        def __getitem__(self, args: int) -> Network: ...
        def __iter__(self) -> Iterator[Network]: ...
        @property
        def max_s_mag(self) -> Network: ...
        def plot_s_smith(self, **kwargs: Any) -> None: ...
        def plot_s_vswr(self, **kwargs: Any) -> None: ...

else:
    from skrf import Frequency, Network, NetworkSet


ArchParams = Tuple[float, float]

Toleranced = tuple[float, float]
ComponentList = Annotated[NDArray[np.float_], Literal["N", 2]]
