"""Constants for the SaySo integration."""

from homeassistant.const import CONF_LLM_HASS_API, CONF_MODEL, CONF_PROMPT

DOMAIN = "sayso"

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_OUTPUT_TOKENS = 160
DEFAULT_MAX_TOOL_ITERATIONS = 3
DEFAULT_TIMEOUT = 30

# Backends. "embedded" runs the GGUF in-process; "external" keeps the original
# OpenAI-compatible HTTP path as an advanced fallback.
BACKEND_EMBEDDED = "embedded"
BACKEND_EXTERNAL = "external"
CONF_BACKEND = "backend"
DEFAULT_BACKEND = BACKEND_EMBEDDED

CONF_MODEL_PATH = "model_path"
CONF_N_THREADS = "n_threads"
CONF_N_CTX = "n_ctx"

# llama-cpp-python publishes sdist only to PyPI and the Home Assistant container
# has no compiler, so this cannot be a manifest requirement. The maintainer's
# index carries musllinux_1_2 wheels for both HAOS architectures.
# See docs/PLAN_EMBEDDED_INFERENCE.md §1.
LLAMA_CPP_PACKAGE = "llama-cpp-python"
LLAMA_CPP_MIN_VERSION = "0.3.33"
LLAMA_CPP_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cpu"

# Default weights: the SaySo Gauntlet v1 fine-tune, published as a GitHub
# Release asset because every useful quant exceeds GitHub's 100 MB file limit.
# Swap these three together to ship a new model; scripts/publish_model.sh
# uploads the asset and rewrites them in one step.
DEFAULT_MODEL_URL = (
    "https://github.com/mossipcams/SaySo/releases/download/model-v1/"
    "SaySo-Gauntlet-v1-Q8_0.gguf"
)
DEFAULT_MODEL_FILENAME = "SaySo-Gauntlet-v1-Q8_0.gguf"
DEFAULT_MODEL_SHA256: str | None = "229c805d85e7ef807bf895d91bf653079ec1eff7ee8d18618baefd6cb4e536f1"

MODEL_STORAGE_SUBDIR = "sayso/models"

# A 230M-class model is prompt-bound on CPU. 4096 holds the system prompt, the
# filtered tool schema, and a few turns without spilling.
DEFAULT_N_CTX = 4096
# Home Assistant shares the box; do not take every core.
MAX_DEFAULT_THREADS = 4

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
ERROR_MODEL_NOT_LOADED = "The local model is not loaded."
ERROR_REQUEST_TIMEOUT = "That request took too long."
ERROR_EMPTY_RESPONSE = "I didn't get a response from the local model."
ERROR_ACTION_FAILED = "I couldn't complete that action."
ERROR_TOOL_ITERATION_LIMIT = "I couldn't complete that action."

CHAT_COMPLETIONS_PATH = "/chat/completions"
MODELS_PATH = "/models"

OPTION_KEYS = (
    CONF_MODEL,
    CONF_MODEL_PATH,
    CONF_N_THREADS,
    CONF_N_CTX,
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
