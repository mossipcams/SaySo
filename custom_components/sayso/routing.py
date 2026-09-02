"""Conservative command-domain routing hints from HA registry metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_CONTROL_VERBS = frozenset(
    {
        "activate",
        "arm",
        "brighten",
        "close",
        "decrease",
        "dim",
        "disable",
        "disarm",
        "enable",
        "flip",
        "increase",
        "lock",
        "lower",
        "off",
        "on",
        "open",
        "pause",
        "play",
        "raise",
        "set",
        "start",
        "stop",
        "switch",
        "toggle",
        "turn",
        "unlock",
    }
)


@dataclass(frozen=True, slots=True)
class RoutingEntity:
    """One exposed entity used for routing hints."""

    entity_id: str
    domain: str
    name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RoutingCatalog:
    """HA-provided entity names and domains for routing."""

    entities: tuple[RoutingEntity, ...]

    @property
    def domains(self) -> frozenset[str]:
        return frozenset(entity.domain for entity in self.entities)


def _normalize_text(text: str) -> str:
    lowered = text.casefold()
    return " ".join(_TOKEN_RE.findall(lowered))


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def _singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 3 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _token_matches_domain(token: str, domain: str) -> bool:
    if token == domain:
        return True
    if token in {f"{domain}s", f"{domain}es"}:
        return True
    return _singularize(token) == domain


def _has_control_verb(command_tokens: list[str]) -> bool:
    return any(token in _CONTROL_VERBS for token in command_tokens)


def _phrase_in_tokens(phrase_tokens: list[str], command_tokens: list[str]) -> bool:
    if not phrase_tokens:
        return False
    width = len(phrase_tokens)
    for index in range(len(command_tokens) - width + 1):
        if command_tokens[index : index + width] == phrase_tokens:
            return True
    return False


def _entity_phrase_tokens(entity: RoutingEntity) -> list[list[str]]:
    phrases = (entity.name, *entity.aliases)
    return [_tokenize(_normalize_text(phrase)) for phrase in phrases if phrase.strip()]


def _domains_from_entity_matches(
    catalog: RoutingCatalog,
    command_tokens: list[str],
) -> set[str]:
    matched: set[str] = set()
    for entity in catalog.entities:
        for phrase_tokens in _entity_phrase_tokens(entity):
            if _phrase_in_tokens(phrase_tokens, command_tokens):
                matched.add(entity.domain)
                break
    return matched


def _domains_from_domain_terms(
    catalog: RoutingCatalog,
    command_tokens: list[str],
) -> set[str]:
    matched: set[str] = set()
    for domain in catalog.domains:
        for token in command_tokens:
            if _token_matches_domain(token, domain):
                matched.add(domain)
                break
    return matched


def identify_command_domain(command: str, catalog: RoutingCatalog) -> str | None:
    """Return a domain hint only for exact, unambiguous token matches."""
    normalized = _normalize_text(command)
    if not normalized or not catalog.entities:
        return None

    command_tokens = _tokenize(normalized)
    if not _has_control_verb(command_tokens):
        return None

    matched_domains = _domains_from_entity_matches(
        catalog,
        command_tokens,
    ) | _domains_from_domain_terms(catalog, command_tokens)

    if len(matched_domains) == 1:
        return next(iter(matched_domains))
    return None
