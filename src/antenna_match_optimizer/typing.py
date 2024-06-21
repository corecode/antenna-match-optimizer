from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Iterator,
    Literal,
    Self,
    Tuple,
)

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:

    class Frequency:
        f: NDArray[np.float_]
        npoints: int
        start: float
        start_scaled: float
        stop: float
        stop_scaled: float
        unit: str

        def __len__(self) -> int: ...
        def __getitem__(self, *args: Any) -> Self: ...

    class Network:
        name: str
        params: dict[Any, Any]
        frequency: Frequency
        s_mag: NDArray[np.float_]
        number_of_ports: int

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
    from skrf import Frequency as Frequency
    from skrf import Network as Network
    from skrf import NetworkSet as NetworkSet


ArchParams = Tuple[float, float]

Toleranced = tuple[float, float]
ComponentList = Annotated[NDArray[np.float_], Literal["N", 2]]
