"""SaySo server package."""

from sayso_server.api import API_VERSION, PROTOCOL_NAME
from sayso_server.auth import bearer_token_valid
from sayso_server.const import WS_PATH
from sayso_server.gateway import handle_ha_connection
from sayso_server.graph_store import HomeGraphStore
from sayso_server.session import HaSession
from sayso_server.conversation import (
    ConversationReferent,
    ConversationStore,
    LastIntent,
    LastTarget,
    ReferentKind,
    SatelliteConversationState,
)
from sayso_server.control_plan import (
    ActionPlan,
    ClarificationPlan,
    ControlPlan,
    NoActionPlan,
    QueryPlan,
    UnsupportedPlan,
)
from sayso_server.envelope import SaySoEnvelope
from sayso_server.home_graph import (
    Area,
    Capability,
    CapabilityKind,
    Device,
    Entity,
    Floor,
    HomeGraphSnapshot,
    Scene,
    Script,
    State,
)
from sayso_server.messages import MESSAGE_TYPES_V1, MessageType
from sayso_server.prompt import PromptOrigin, build_lfm_prompt
from sayso_server.parser import parse_model_output
from sayso_server.protocol import parse_envelope, parse_envelope_json
from sayso_server.runtime import FakeModelRuntime, ModelMetadata, ModelRuntime, PlanGenerationResult
from sayso_server.schema import control_plan_json_schema, sayso_api_v1_json_schema

__all__ = [
    "API_VERSION",
    "ActionPlan",
    "Area",
    "Capability",
    "CapabilityKind",
    "ClarificationPlan",
    "ControlPlan",
    "ConversationReferent",
    "ConversationStore",
    "Device",
    "Entity",
    "FakeModelRuntime",
    "HaSession",
    "HomeGraphStore",
    "WS_PATH",
    "bearer_token_valid",
    "handle_ha_connection",
    "Floor",
    "HomeGraphSnapshot",
    "LastIntent",
    "LastTarget",
    "MESSAGE_TYPES_V1",
    "MessageType",
    "ModelMetadata",
    "ModelRuntime",
    "NoActionPlan",
    "PlanGenerationResult",
    "PromptOrigin",
    "PROTOCOL_NAME",
    "build_lfm_prompt",
    "QueryPlan",
    "ReferentKind",
    "SaySoEnvelope",
    "SatelliteConversationState",
    "Scene",
    "Script",
    "State",
    "UnsupportedPlan",
    "__version__",
    "control_plan_json_schema",
    "parse_envelope",
    "parse_envelope_json",
    "parse_model_output",
    "sayso_api_v1_json_schema",
]
__version__ = "0.1.0"
