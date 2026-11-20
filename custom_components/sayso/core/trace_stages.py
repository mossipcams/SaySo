from enum import StrEnum


class Stage(StrEnum):

    AUDIO_UPLOAD = "audio_upload"
    STT = "stt"
    SAYSO_REQUEST = "sayso_request"
    CONTEXT = "context"
    INFERENCE = "inference"
    TOOL_PARSE = "tool_parse"
    HA_ACTION = "ha_action"
    RESPONSE = "response"
    TTS = "tts"
    PLAYBACK = "playback"


class ErrorType(StrEnum):

    STT_FAILED = "stt_failed"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_TIMEOUT = "model_timeout"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    EMPTY_RESPONSE = "empty_response"
    SCHEMA_MISMATCH = "schema_mismatch"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNAVAILABLE_TOOL = "unavailable_tool"
    ITERATION_LIMIT = "iteration_limit"
    HA_ACTION_FAILED = "ha_action_failed"
    TTS_FAILED = "tts_failed"
    PLAYBACK_FAILED = "playback_failed"
    PIPELINE_ERROR = "pipeline_error"
    UNKNOWN = "unknown"
