"""Mnemoverse long-term memory for Microsoft Agent Framework (Python).

* :class:`MnemoverseContextProvider` recalls from the user's memory before each run.
* :func:`mnemoverse_tools` gives the agent search, save and rate tools.
* :func:`bind_user` and :func:`bind_domain` name the domain a session reads and writes.
"""

from ._common import bind_domain, bind_user, domain_of
from ._provider import MnemoverseContextProvider
from ._tools import mnemoverse_tools
from ._version import __version__

__all__ = [
    "MnemoverseContextProvider",
    "__version__",
    "bind_domain",
    "bind_user",
    "domain_of",
    "mnemoverse_tools",
]
