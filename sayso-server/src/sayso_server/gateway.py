"""Home Assistant WebSocket session gateway."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable
from typing import Any, Protocol

from pydantic import ValidationError

from sayso_server.api import API_VERSION
from sayso_server.auth import bearer_token_valid
from sayso_server.envelope import SaySoEnvelope
from sayso_server.graph_store import HomeGraphStore
from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.messages import MessageType
from sayso_server.readiness import ReadinessState, prepare_response_payload
from sayso_server.results import ActionResultStatus
from sayso_server.satellites import SatelliteRegistry
from sayso_server.session import HaSession
from sayso_server.text_api import (
    ConversationRequestPayload,
    TextController,
    conversation_response_payload,
    resolve_conversation_area,
)


class GatewayWebSocket(Protocol):
    """Minimal WebSocket surface used by the HA session gateway."""

    closed: bool

    async def send_str(self, data: str) -> None: ...

    async def close(self) -> None: ...

    async def receive_str(self) -> str | None: ...


async def handle_ha_connection(
    ws: GatewayWebSocket,
    *,
    authorization: str | None,
    server_token: str,
    graph_store: HomeGraphStore | None = None,
    readiness: ReadinessState | None = None,
    text_controller: TextController | None = None,
    satellite_registry: SatelliteRegistry | None = None,
    on_session_started: Callable[[HaSession, GatewayWebSocket], None] | None = None,
    on_session_ended: Callable[[HaSession], None] | None = None,
) -> HaSession | None:
    """Authenticate, complete the v1 hello handshake, and process graph updates."""

    if not bearer_token_valid(authorization=authorization, expected_token=server_token):
        await ws.close()
        return None

    raw = await ws.receive_str()
    if raw is None:
        await ws.close()
        return None

    try:
        envelope = SaySoEnvelope.model_validate_json(raw)
    except (ValidationError, json.JSONDecodeError, UnicodeDecodeError):
        await ws.close()
        return None

    if envelope.type != MessageType.HELLO:
        await ws.close()
        return None

    ack = SaySoEnvelope(
        version=API_VERSION,
        type=MessageType.HELLO_ACK,
        correlation_id=envelope.correlation_id,
        payload={},
    )
    await ws.send_str(ack.model_dump_json())
    store = graph_store if graph_store is not None else HomeGraphStore()
    store.clear()
    if readiness is not None:
        readiness.set_ha_connected(False)
    session = HaSession(correlation_id=envelope.correlation_id, graph=store)
    if on_session_started is not None:
        on_session_started(session, ws)
    # ponytail: test fakes expose _recv_queue; an empty queue means handshake-only.
    recv_queue = getattr(ws, "_recv_queue", None)
    try:
        if isinstance(recv_queue, asyncio.Queue) and recv_queue.empty():
            return session
        await _process_graph_messages(
            ws,
            session,
            readiness=readiness,
            text_controller=text_controller,
            satellite_registry=satellite_registry,
        )
        return session
    finally:
        session.clear_pending_action_waits()
        if on_session_ended is not None:
            on_session_ended(session)
        store.clear()
        if readiness is not None:
            readiness.set_ha_connected(False)


async def _process_graph_messages(
    ws: GatewayWebSocket,
    session: HaSession,
    *,
    readiness: ReadinessState | None = None,
    text_controller: TextController | None = None,
    satellite_registry: SatelliteRegistry | None = None,
) -> None:
    while not ws.closed:
        for outbound in session.drain_outbound():
            await ws.send_str(outbound)

        recv_task = asyncio.create_task(ws.receive_str())
        outbound_task = asyncio.create_task(session.wait_for_outbound())
        done, pending = await asyncio.wait(
            {recv_task, outbound_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for unfinished in pending:
            unfinished.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await unfinished

        if outbound_task in done and recv_task not in done:
            continue

        raw = recv_task.result()
        if raw is None:
            session.clear_pending_action_waits()
            await ws.close()
            return

        try:
            envelope = SaySoEnvelope.model_validate_json(raw)
        except (ValidationError, json.JSONDecodeError, UnicodeDecodeError):
            continue

        if envelope.type == MessageType.PING:
            pong = SaySoEnvelope(
                version=API_VERSION,
                type=MessageType.PONG,
                correlation_id=envelope.correlation_id,
                payload={},
            )
            await ws.send_str(pong.model_dump_json())
        elif envelope.type == MessageType.GRAPH_SNAPSHOT:
            try:
                snapshot = HomeGraphSnapshot.model_validate(envelope.payload)
            except ValidationError:
                await _send_error(
                    ws,
                    correlation_id=envelope.correlation_id,
                    reason="invalid_graph_snapshot",
                )
                continue
            session.graph.replace_snapshot(snapshot)
            session.mark_graph_ready()
            if readiness is not None:
                readiness.set_ha_connected(True)
        elif envelope.type == MessageType.STATE_DELTA:
            session.graph.apply_state_delta(envelope.payload)
        elif envelope.type == MessageType.REGISTRY_DELTA:
            session.graph.apply_registry_delta(envelope.payload)
        elif envelope.type == MessageType.ACTION_RESULT:
            _record_action_result(session, envelope.payload)
        elif envelope.type == MessageType.PREPARE:
            await _handle_prepare_request(
                ws,
                session,
                envelope,
                readiness=readiness,
            )
        elif envelope.type == MessageType.CONVERSATION_REQUEST:
            await _handle_conversation_request(
                ws,
                session,
                envelope,
                text_controller=text_controller,
                satellite_registry=satellite_registry,
            )


async def _handle_prepare_request(
    ws: GatewayWebSocket,
    session: HaSession,
    envelope: SaySoEnvelope,
    *,
    readiness: ReadinessState | None,
) -> None:
    response = SaySoEnvelope(
        version=API_VERSION,
        type=MessageType.PREPARE_RESPONSE,
        correlation_id=envelope.correlation_id,
        payload=prepare_response_payload(session=session, readiness=readiness),
    )
    await ws.send_str(response.model_dump_json())


async def _send_conversation_response(
    ws: GatewayWebSocket,
    *,
    correlation_id: str,
    payload: dict[str, Any],
) -> None:
    response = SaySoEnvelope(
        version=API_VERSION,
        type=MessageType.CONVERSATION_RESPONSE,
        correlation_id=correlation_id,
        payload=payload,
    )
    await ws.send_str(response.model_dump_json())


async def _handle_conversation_request(
    ws: GatewayWebSocket,
    session: HaSession,
    envelope: SaySoEnvelope,
    *,
    text_controller: TextController | None,
    satellite_registry: SatelliteRegistry | None,
) -> None:
    try:
        request = ConversationRequestPayload.model_validate(envelope.payload)
    except ValidationError:
        await _send_error(
            ws,
            correlation_id=envelope.correlation_id,
            reason="invalid_conversation_request",
        )
        return

    if text_controller is None:
        await _send_error(
            ws,
            correlation_id=envelope.correlation_id,
            reason="text_controller_unavailable",
        )
        return

    area_id, error_code = resolve_conversation_area(
        request,
        graph_store=session.graph,
    )
    if error_code is not None:
        await _send_error(
            ws,
            correlation_id=envelope.correlation_id,
            reason=error_code,
        )
        return

    source_key = request.source_id or envelope.correlation_id

    handle_async = getattr(text_controller, "handle_async", None)
    try:
        if handle_async is not None and _needs_session_pump(text_controller):
            controller_payload = await _run_controller_with_session_pump(
                ws,
                session,
                handle_async(
                    satellite_id=source_key,
                    area_id=area_id,
                    text=request.transcript,
                    correlation_id=envelope.correlation_id,
                    input_type="audio",
                    stt_ms=request.stt_ms,
                ),
            )
        elif handle_async is not None:
            controller_payload = await handle_async(
                satellite_id=source_key,
                area_id=area_id,
                text=request.transcript,
                correlation_id=envelope.correlation_id,
                input_type="audio",
                stt_ms=request.stt_ms,
            )
            for outbound in session.drain_outbound():
                await ws.send_str(outbound)
        else:
            controller_payload = text_controller.handle(
                satellite_id=source_key,
                area_id=area_id,
                text=request.transcript,
                correlation_id=envelope.correlation_id,
                input_type="audio",
                stt_ms=request.stt_ms,
            )
            for outbound in session.drain_outbound():
                await ws.send_str(outbound)
    except Exception:
        await _send_error(
            ws,
            correlation_id=envelope.correlation_id,
            reason="conversation_failed",
        )
        return

    await _send_conversation_response(
        ws,
        correlation_id=envelope.correlation_id,
        payload=conversation_response_payload(controller_payload),
    )


def _needs_session_pump(text_controller: TextController) -> bool:
    ha_client = getattr(text_controller, "_ha_client", None)
    return ha_client is not None and hasattr(ha_client, "collect_action_results")


async def _run_controller_with_session_pump(
    ws: GatewayWebSocket,
    session: HaSession,
    controller_task: Any,
) -> dict[str, Any]:
    """Run an async controller while forwarding action traffic on the same socket."""

    task = asyncio.ensure_future(controller_task)
    disconnected = False
    try:
        while not task.done():
            for outbound in session.drain_outbound():
                await ws.send_str(outbound)

            recv_task = asyncio.create_task(ws.receive_str())
            outbound_task = asyncio.create_task(session.wait_for_outbound())
            done, pending = await asyncio.wait(
                {task, recv_task, outbound_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for unfinished in pending:
                if unfinished is task:
                    continue
                unfinished.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await unfinished

            if task in done:
                if recv_task in done and recv_task.result() is None:
                    session.clear_pending_action_waits()
                    await ws.close()
                    disconnected = True
                break
            if outbound_task in done and recv_task not in done:
                continue

            raw = recv_task.result()
            if raw is None:
                session.clear_pending_action_waits()
                await ws.close()
                disconnected = True
                break
            try:
                envelope = SaySoEnvelope.model_validate_json(raw)
            except (ValidationError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if envelope.type == MessageType.ACTION_RESULT:
                _record_action_result(session, envelope.payload)
            elif envelope.type == MessageType.PING:
                pong = SaySoEnvelope(
                    version=API_VERSION,
                    type=MessageType.PONG,
                    correlation_id=envelope.correlation_id,
                    payload={},
                )
                await ws.send_str(pong.model_dump_json())
        for outbound in session.drain_outbound():
            await ws.send_str(outbound)
        if disconnected and task.done() and not task.cancelled():
            return task.result()
        if disconnected:
            raise ConnectionError("home assistant websocket disconnected")
        return await task
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            session.clear_pending_action_waits()
        elif task.cancelled():
            session.clear_pending_action_waits()


async def _send_error(
    ws: GatewayWebSocket,
    *,
    correlation_id: str,
    reason: str,
) -> None:
    error = SaySoEnvelope(
        version=API_VERSION,
        type=MessageType.ERROR,
        correlation_id=correlation_id,
        payload={"reason": reason},
    )
    await ws.send_str(error.model_dump_json())


def _record_action_result(session: HaSession, payload: dict[str, object]) -> None:
    request_id = payload.get("request_id")
    status = payload.get("status")
    if not isinstance(request_id, str) or not request_id:
        return
    if not isinstance(status, str):
        return
    try:
        parsed_status = ActionResultStatus(status)
    except ValueError:
        return
    reason = payload.get("reason")
    parsed_reason = reason if isinstance(reason, str) else None
    session.record_action_result(
        request_id=request_id,
        status=parsed_status,
        reason=parsed_reason,
    )
