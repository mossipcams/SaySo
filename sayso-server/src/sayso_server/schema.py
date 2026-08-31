"""JSON Schema helpers for SaySo contracts."""

from sayso_server.control_plan import ControlPlan
from sayso_server.envelope import SaySoEnvelope


def control_plan_json_schema() -> dict[str, object]:
    return ControlPlan.json_schema()


def sayso_api_v1_json_schema() -> dict[str, object]:
    return SaySoEnvelope.json_schema()
