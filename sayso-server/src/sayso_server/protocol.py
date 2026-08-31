"""Parse and validate SaySo API v1 wire messages."""

from sayso_server.envelope import SaySoEnvelope


def parse_envelope(data: object) -> SaySoEnvelope:
    return SaySoEnvelope.model_validate(data)


def parse_envelope_json(json_data: str | bytes | bytearray) -> SaySoEnvelope:
    return SaySoEnvelope.model_validate_json(json_data)
