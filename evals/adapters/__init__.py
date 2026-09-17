"""Model adapters. They differ only in how they invoke the model."""

from evals.adapters.endpoint import EndpointAdapter
from evals.adapters.in_memory import InMemoryAdapter

__all__ = ["EndpointAdapter", "InMemoryAdapter"]
