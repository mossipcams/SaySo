"""Constants for the SaySo integration."""

from homeassistant.const import CONF_LLM_HASS_API, CONF_MODEL, CONF_PROMPT

DOMAIN = "sayso"

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_OUTPUT_TOKENS = 160
DEFAULT_MAX_TOOL_ITERATIONS = 3
DEFAULT_TIMEOUT = 30

CONF_TIMEOUT = "timeout"
CONF_MAX_OUTPUT_TOKENS = "max_output_tokens"
CONF_MAX_TOOL_ITERATIONS = "max_tool_iterations"
CONF_TEMPERATURE = "temperature"
CONF_TRACE_RETENTION_DAYS = "trace_retention_days"
CONF_TRACE_MAX_INTERACTIONS = "trace_max_interactions"
CONF_TRACE_STORE_UTTERANCES = "trace_store_utterances"

# Retention. Whichever limit is reached first applies. The interaction cap is
# the one that fits the JSON store cleanly; 30 days is the conservative age.
DEFAULT_TRACE_RETENTION_DAYS = 30
DEFAULT_TRACE_MAX_INTERACTIONS = 500
MAX_TRACE_INTERACTIONS = 5000
DEFAULT_TRACE_STORE_UTTERANCES = True

DEFAULT_SYSTEM_PROMPT = """You are SaySo, a local Home Assistant voice agent.
Use the available tools for home state queries and actions. Only claim an action succeeded when its tool result confirms success. Use names, areas, and context supplied by Home Assistant. If a request is ambiguous, ask one short question. Keep spoken responses brief. Do not describe tool calls."""

ERROR_MODEL_UNAVAILABLE = "The local model is unavailable."
ERROR_REQUEST_TIMEOUT = "That request took too long."
ERROR_EMPTY_RESPONSE = "I didn't get a response from the local model."
ERROR_ACTION_FAILED = "I couldn't complete that action."
ERROR_TOOL_ITERATION_LIMIT = "I couldn't complete that action."

CHAT_COMPLETIONS_PATH = "/chat/completions"
MODELS_PATH = "/models"

OPTION_KEYS = (
    CONF_MODEL,
    CONF_TIMEOUT,
    CONF_LLM_HASS_API,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_MAX_OUTPUT_TOKENS,
    CONF_MAX_TOOL_ITERATIONS,
    CONF_TRACE_RETENTION_DAYS,
    CONF_TRACE_MAX_INTERACTIONS,
    CONF_TRACE_STORE_UTTERANCES,
)

SERVICE_GET_TRACE = "get_trace"
SERVICE_LIST_TRACES = "list_traces"
