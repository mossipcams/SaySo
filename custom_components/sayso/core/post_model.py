from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from .completion import ChatCompletionResult, ToolCall
from .correction import build_correction_messages
from .exceptions import SaySoError, SaySoTimeoutError
from .grounding import (
    GroundingEntity,
    GroundingHome,
    clarify,
    drop_unspoken_area,
    is_grounded,
)
from .held import HeldState, is_cancel, is_expired, resolve_held_option
from .tool_contract import (
    ToolArgumentFailureCode,
    ToolArgumentValidationError,
    contract_failures,
    is_well_formed_batch,
    tools_by_name,
)
from .trace_stages import Stage
from .vocabulary import DEFAULT_LANGUAGE, SPEECH_EN, Speech, Vocabulary, vocabulary_for


class Phase(StrEnum):
    INITIAL = "initial"
    CORRECTION = "correction"
    FOLLOW_UP = "follow_up"
    EXECUTION = "execution"


class FailureKind(StrEnum):
    SCHEMA_MISMATCH = "schema_mismatch"
    INVALID_ARGUMENTS = "invalid_arguments"
    UNAVAILABLE_TOOL = "unavailable_tool"
    NO_TOOL_API = "no_tool_api"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    ITERATION_LIMIT = "iteration_limit"
    EMPTY_CORRECTION = "empty_correction"
    EMPTY_RESPONSE = "empty_response"
    TIMEOUT = "timeout"
    MODEL_FAILED = "model_failed"


class Recorder(Protocol):
    def open(self, stage: Stage | str, **metadata: Any) -> Any: ...

    def close(self, span: Any, *, success: bool = True) -> None: ...


@dataclass(frozen=True, slots=True)
class TurnInput:
    user_text: str
    transcript: Callable[[], list[dict[str, Any]]]
    tools: tuple[dict[str, Any], ...]
    schema_fingerprint: str | None
    available_names: set[str]
    action_tools: frozenset[str]
    exposed_domains: frozenset[str]
    home: Callable[[], GroundingHome] | None
    tool_api: bool
    iteration_limit: Callable[[], int]
    now: float
    held_ttl: float
    held: HeldState | None = None
    calls: tuple[ToolCall, ...] | None = None
    confirmed: bool = False
    read_only: frozenset[str] = frozenset()
    language: str = DEFAULT_LANGUAGE
    speech: Speech = SPEECH_EN


@dataclass(frozen=True, slots=True)
class ValidationResult:
    validated: list[tuple[ToolCall, dict[str, Any]]]
    failures: list[tuple[ToolCall, ToolArgumentValidationError]]


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    failed: bool
    ha_error: str | None = None
    ha_error_text: str | None = None


@dataclass(frozen=True, slots=True)
class Failure:
    kind: FailureKind
    phase: Phase
    error: BaseException | None = None
    validated: tuple[tuple[ToolCall, dict[str, Any]], ...] = ()
    ha_error: str | None = None
    ha_error_text: str | None = None


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    speech: str | None = None
    failure: Failure | None = None
    held: HeldState | None = None
    correction_used: bool = False
    executed: tuple[ToolCall, ...] = ()
    clarified: bool = False
    done_shortcut: bool = False
    area_dropped: bool = False
    held_replayed: bool = False
    first: ChatCompletionResult | None = None


Complete = Callable[..., Awaitable[ChatCompletionResult]]
Execute = Callable[[list[tuple[ToolCall, dict[str, Any]]]], Awaitable[ExecutionResult]]
Validate = Callable[[list[ToolCall]], ValidationResult]


@dataclass(slots=True)
class _Run:
    turn: TurnInput
    complete: Complete
    validate: Validate
    execute: Execute
    recorder: Recorder
    phase: Phase = Phase.INITIAL
    correction_used: bool = False
    tools_executed: bool = False
    iteration: int = 1
    executed: list[ToolCall] = field(default_factory=list)
    clarified: bool = False
    done_shortcut: bool = False
    area_dropped: bool = False
    held_replayed: bool = False
    first: ChatCompletionResult | None = None

    @property
    def vocab(self) -> Vocabulary:
        return vocabulary_for(self.turn.language)

    def outcome(
        self, *, speech: str | None = None, held: HeldState | None = None
    ) -> TurnOutcome:
        return TurnOutcome(
            speech=speech,
            held=held,
            correction_used=self.correction_used,
            executed=tuple(self.executed),
            clarified=self.clarified,
            done_shortcut=self.done_shortcut,
            area_dropped=self.area_dropped,
            held_replayed=self.held_replayed,
            first=self.first,
        )

    async def _complete(
        self, messages: list[dict[str, Any]], *, correction: bool
    ) -> ChatCompletionResult:
        result = await self.complete(messages, correction=correction)
        if self.first is None:
            self.first = result
        return result

    def fail(
        self,
        kind: FailureKind,
        *,
        phase: Phase | None = None,
        error: BaseException | None = None,
        validated: tuple[tuple[ToolCall, dict[str, Any]], ...] = (),
        ha_error: str | None = None,
        ha_error_text: str | None = None,
    ) -> TurnOutcome:
        return TurnOutcome(
            failure=Failure(
                kind,
                self.phase if phase is None else phase,
                error=error,
                validated=validated,
                ha_error=ha_error,
                ha_error_text=ha_error_text,
            ),
            correction_used=self.correction_used,
            executed=tuple(self.executed),
            clarified=self.clarified,
            done_shortcut=self.done_shortcut,
            area_dropped=self.area_dropped,
            held_replayed=self.held_replayed,
            first=self.first,
        )

    async def run(self) -> TurnOutcome:
        turn = self.turn
        if turn.calls is not None:
            return await self.tool_loop(
                list(turn.calls), exempt=None, grounding_text=turn.user_text
            )
        if (
            turn.held is not None
            and turn.home is not None
            and not is_expired(turn.held, turn.now)
        ):
            if is_cancel(turn.user_text, self.vocab):
                return self.outcome(speech=turn.speech.cancelled)
            option = resolve_held_option(turn.held, turn.user_text, self.vocab)
            if option is not None:
                self.held_replayed = True
                return await self.tool_loop(
                    _replay_calls(turn.held, option),
                    exempt=turn.held.index,
                    grounding_text=turn.held.utterance,
                )
        messages = turn.transcript()
        try:
            result = await self._complete(messages, correction=False)
        except SaySoError as err:
            return self.fail(_model_kind(err), error=err)
        if result.tool_calls:
            return await self.tool_loop(
                result.tool_calls, exempt=None, grounding_text=turn.user_text
            )
        if not (result.content or "").strip():
            return self.fail(FailureKind.EMPTY_RESPONSE)
        return self.outcome(speech=result.content)

    async def tool_loop(
        self,
        current: list[ToolCall],
        *,
        exempt: int | None,
        grounding_text: str,
    ) -> TurnOutcome:
        turn = self.turn
        if not turn.tool_api:
            return self.fail(FailureKind.NO_TOOL_API)
        compiled = tools_by_name(turn.tools)
        while True:
            parse_span = self.recorder.open(
                Stage.TOOL_PARSE, **{"sayso.calls": len(current)}
            )

            if not is_well_formed_batch(current):
                return self.fail(FailureKind.INVALID_ARGUMENTS)
            if any(call.name not in turn.available_names for call in current):
                return self.fail(FailureKind.UNAVAILABLE_TOOL)

            if failures := contract_failures(current, compiled, turn.exposed_domains):
                validated = []
            else:
                validation = self.validate(current)
                validated, failures = validation.validated, validation.failures

            if failures:
                repairable = not (
                    validated or self.correction_used or self.tools_executed
                )
                if not repairable or turn.schema_fingerprint is None:
                    return self.fail(_validation_kind(failures[0][1]))

                self.phase = Phase.CORRECTION
                self.recorder.close(parse_span, success=False)
                messages = build_correction_messages(
                    turn.transcript(),
                    failures,
                    turn.available_names,
                    turn.schema_fingerprint,
                )
                try:
                    corrected = await self._complete(messages, correction=True)
                except SaySoError as err:
                    return self.fail(_model_kind(err), error=err)

                self.correction_used = True
                if not corrected.tool_calls:
                    return self.fail(FailureKind.EMPTY_CORRECTION)
                current = corrected.tool_calls
                exempt = None
                continue

            if turn.home is not None:
                home = turn.home()
                normalized: list[tuple[ToolCall, dict[str, Any]]] = []
                for i, (call, args) in enumerate(validated):
                    if call.name in turn.read_only or i == exempt:
                        normalized.append((call, args))
                        continue
                    new_args = drop_unspoken_area(call.name, args, grounding_text, home)
                    if new_args is not args:
                        self.area_dropped = True
                        parse_span.metadata["sayso.area_dropped"] = {
                            "tool": call.name,
                            "name": args.get("name"),
                            "area": args.get("area"),
                        }
                    normalized.append((call, new_args))
                validated = normalized
                for i, (call, args) in enumerate(validated):
                    if call.name in turn.read_only or i == exempt:
                        continue
                    others = [
                        (other.name, other_args)
                        for j, (other, other_args) in enumerate(validated)
                        if j != i
                    ]
                    if not is_grounded(
                        call.name,
                        args,
                        grounding_text,
                        home,
                        others=others,
                        vocab=self.vocab,
                    ):
                        clar = clarify(
                            call.name,
                            args,
                            grounding_text,
                            home,
                            speech=turn.speech,
                            vocab=self.vocab,
                        )
                        parse_span.metadata["sayso.clarify"] = clar.speech
                        self.recorder.close(parse_span, success=False)
                        self.clarified = True
                        held = None
                        if clar.options:
                            held = HeldState(
                                [c for c, _ in validated],
                                i,
                                clar.options,
                                turn.now + turn.held_ttl,
                                grounding_text,
                            )
                        return self.outcome(speech=clar.speech, held=held)
            self.recorder.close(parse_span)

            self.executed.extend(
                ToolCall(id=call.id, name=call.name, arguments=args)
                for call, args in validated
            )
            execution = await self.execute(validated)
            if execution.failed:
                return self.fail(
                    FailureKind.TOOL_EXECUTION_FAILED,
                    phase=Phase.EXECUTION,
                    validated=tuple(validated),
                    ha_error=execution.ha_error,
                    ha_error_text=execution.ha_error_text,
                )

            self.tools_executed = True
            exempt = None
            if len(validated) == 1 and validated[0][0].name in turn.action_tools:
                self.done_shortcut = True
                return self.outcome(speech=turn.speech.done)

            self.phase = Phase.FOLLOW_UP
            try:
                follow_up = await self._complete(turn.transcript(), correction=False)
            except SaySoError as err:
                return self.fail(_model_kind(err), error=err)

            if follow_up.tool_calls:
                if self.iteration >= turn.iteration_limit():
                    return self.fail(FailureKind.ITERATION_LIMIT)
                self.iteration += 1
                current = follow_up.tool_calls
                continue

            if not (follow_up.content or "").strip():
                return self.fail(FailureKind.EMPTY_RESPONSE)
            return self.outcome(speech=follow_up.content)


def _replay_calls(held: HeldState, option: GroundingEntity) -> list[ToolCall]:
    base_args = held.calls[held.index].arguments
    new_args = {
        key: value
        for key, value in base_args.items()
        if key not in {"area", "floor", "domain", "device_class"}
    }
    new_args["name"] = option.name
    calls = list(held.calls)
    held_call = held.calls[held.index]
    calls[held.index] = ToolCall(
        id=held_call.id,
        name=held_call.name,
        arguments=new_args,
    )
    return calls


def _validation_kind(error: ToolArgumentValidationError) -> FailureKind:
    if error.code == ToolArgumentFailureCode.SCHEMA_MISMATCH:
        return FailureKind.SCHEMA_MISMATCH
    return FailureKind.INVALID_ARGUMENTS


def _model_kind(error: SaySoError) -> FailureKind:
    if isinstance(error, SaySoTimeoutError):
        return FailureKind.TIMEOUT
    return FailureKind.MODEL_FAILED


async def run_turn(
    turn: TurnInput,
    *,
    complete: Complete,
    validate: Validate,
    execute: Execute,
    recorder: Recorder,
) -> TurnOutcome:
    return await _Run(
        turn=turn,
        complete=complete,
        validate=validate,
        execute=execute,
        recorder=recorder,
    ).run()
