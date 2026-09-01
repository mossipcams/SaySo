"""SaySo server package."""

from sayso_server.ambiguity import (
    CandidateSelection,
    candidates_within_score_margin,
    is_ambiguous,
    resolve_candidate_selection,
    resolve_candidates_for_request,
)
from sayso_server.api import API_VERSION, PROTOCOL_NAME
from sayso_server.auth import bearer_token_valid
from sayso_server.capability import CapabilityValidationError, validate_target_capabilities
from sayso_server.candidates import CandidateRequest, ScoredCandidate, retrieve_candidates
from sayso_server.const import TEXT_PATH, WS_PATH
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
from sayso_server.gateway import handle_ha_connection
from sayso_server.graph_store import HomeGraphStore
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
from sayso_server.normalize import normalize_tokens
from sayso_server.parser import parse_model_output
from sayso_server.prompt import PromptOrigin, build_lfm_prompt
from sayso_server.protocol import parse_envelope, parse_envelope_json
from sayso_server.exclusions import (
    apply_inclusions_exclusions,
    filter_entity_ids_by_domain,
    matches_semantic_name,
    resolve_names_in_scope,
)
from sayso_server.followups import FollowUpResolution, is_follow_up_intent, resolve_follow_up
from sayso_server.resolver import resolve_entity_ids
from sayso_server.ha_client import ActionRequest, ActionRequestClient, FakeHaClient, HaClient, ServiceCall
from sayso_server.orchestrator import classify_action_results, execute_control_plan
from sayso_server.queries import QueryOutcome, evaluate_query
from sayso_server.results import ActionResult, ActionResultStatus, ExecutionCategory, ExecutionOutcome
from sayso_server.runtime import FakeModelRuntime, ModelMetadata, ModelRuntime, PlanGenerationResult
from sayso_server.safety import evaluate_safety_barrier, execute_if_safe
from sayso_server.scope import expand_scope
from sayso_server.schema import control_plan_json_schema, sayso_api_v1_json_schema
from sayso_server.satellites import SatelliteRegistry
from sayso_server.text_api import (
    OrchestratorTextController,
    TextController,
    TextRequestEnvelope,
    TextResponseEnvelope,
    create_text_handler,
)
from sayso_server.scoring import DEFAULT_AMBIGUITY_MARGIN, ScoreBreakdown
from sayso_server.session import HaSession

__all__ = [
    "API_VERSION",
    "ActionPlan",
    "ActionRequest",
    "ActionRequestClient",
    "ActionResult",
    "ActionResultStatus",
    "Area",
    "Capability",
    "CapabilityKind",
    "CandidateRequest",
    "CandidateSelection",
    "CapabilityValidationError",
    "ClarificationPlan",
    "ControlPlan",
    "DEFAULT_AMBIGUITY_MARGIN",
    "ConversationReferent",
    "ConversationStore",
    "Device",
    "Entity",
    "FakeHaClient",
    "FakeModelRuntime",
    "FollowUpResolution",
    "HaClient",
    "HaSession",
    "HomeGraphStore",
    "SatelliteRegistry",
    "TEXT_PATH",
    "TextController",
    "TextRequestEnvelope",
    "TextResponseEnvelope",
    "OrchestratorTextController",
    "create_text_handler",
    "WS_PATH",
    "bearer_token_valid",
    "handle_ha_connection",
    "is_ambiguous",
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
    "QueryOutcome",
    "QueryPlan",
    "ReferentKind",
    "SaySoEnvelope",
    "SatelliteConversationState",
    "Scene",
    "ScoreBreakdown",
    "ScoredCandidate",
    "Script",
    "ServiceCall",
    "State",
    "UnsupportedPlan",
    "__version__",
    "build_lfm_prompt",
    "candidates_within_score_margin",
    "control_plan_json_schema",
    "apply_inclusions_exclusions",
    "expand_scope",
    "filter_entity_ids_by_domain",
    "matches_semantic_name",
    "normalize_tokens",
    "parse_envelope",
    "parse_envelope_json",
    "parse_model_output",
    "resolve_candidate_selection",
    "resolve_candidates_for_request",
    "resolve_entity_ids",
    "resolve_names_in_scope",
    "retrieve_candidates",
    "sayso_api_v1_json_schema",
    "ExecutionCategory",
    "ExecutionOutcome",
    "classify_action_results",
    "evaluate_query",
    "evaluate_safety_barrier",
    "execute_control_plan",
    "execute_if_safe",
    "is_follow_up_intent",
    "resolve_follow_up",
    "validate_target_capabilities",
]
__version__ = "0.1.0"
