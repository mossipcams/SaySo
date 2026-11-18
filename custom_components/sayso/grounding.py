from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .const import (
    DOMAIN_PLURALS,
    ERROR_AREA_HAS_NO_DOMAIN,
    ERROR_CHOOSE,
    ERROR_DID_YOU_MEAN,
    ERROR_NOT_UNDERSTOOD,
)

_TOKEN_RE = re.compile(r"[a-z0-9']+")

GENERIC = {
    "the",
    "tv",
    "light",
    "lights",
    "lamp",
    "switch",
    "plug",
    "fan",
    "speaker",
    "volume",
}

NOUN = {
    "light": "light",
    "lights": "light",
    "lamp": "light",
    "lamps": "light",
    "tv": "media_player",
    "fan": "fan",
    "volume": "media_player",
}

_ALWAYS_GROUNDED = {"GetLiveContext", "GetDateTime"}


@dataclass(frozen=True)
class GroundingEntity:
    entity_id: str
    name: str
    aliases: tuple[str, ...]
    domain: str
    area: str | None
    device_id: str | None


@dataclass(frozen=True)
class GroundingHome:
    entities: tuple[GroundingEntity, ...]
    areas: dict[str, tuple[str, ...]]
    satellite_area: str | None
    satellite_device_id: str | None


@dataclass(frozen=True)
class Clarification:
    speech: str
    options: tuple[GroundingEntity, ...]


def tokens(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.casefold())


def labels(entity: GroundingEntity) -> set[str]:
    values = [entity.name, *entity.aliases]
    return {v.casefold().strip() for v in values}


def word_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def support(label: str, utterance: str, generic_only: float = 1.0) -> float:
    ut = tokens(utterance)
    d = [t for t in tokens(label) if t not in GENERIC]
    u = [t for t in ut if t not in GENERIC and t not in NOUN] or [""]
    if not d:
        return generic_only if any(t in ut for t in tokens(label)) else 0.0
    scores = []
    for t in d:
        best = max(
            (r for r in (word_ratio(t, x) for x in u) if r >= 0.6),
            default=0.0,
        )
        scores.append(best)
    return sum(scores) / len(scores)


def mentioned_area(utterance: str, home: GroundingHome) -> str | None:
    ut = tokens(utterance)
    best: tuple[int, str] | None = None
    for area, aliases in home.areas.items():
        for w in (area.casefold(), *(a.casefold() for a in aliases)):
            ws = [p for p in re.split(r"[^a-z0-9']+", w) if p]
            if not ws:
                continue
            n = len(ws)
            for i, tok in enumerate(ut):
                if ut[i : i + n] == ws and (best is None or n > best[0]):
                    best = (n, area)
    return best[1] if best else None


def _phrase_in_utterance(phrase: str, ut_tokens: list[str]) -> bool:
    ws = [p for p in re.split(r"[^a-z0-9']+", phrase.casefold()) if p]
    if not ws:
        return False
    n = len(ws)
    for i in range(len(ut_tokens) - n + 1):
        if ut_tokens[i : i + n] == ws:
            return True
    return False


def _area_uttered(area: str, utterance: str, home: GroundingHome) -> bool:
    if (
        home.satellite_area is not None
        and area.casefold() == home.satellite_area.casefold()
    ):
        return True
    ut = tokens(utterance)
    names = [area.casefold(), *(a.casefold() for a in home.areas.get(area, ()))]
    return any(_phrase_in_utterance(w, ut) for w in names)


def drop_unspoken_area(
    tool_name: str, args: dict, utterance: str, home: GroundingHome
) -> dict:
    name = args.get("name")
    area = args.get("area")
    if not isinstance(name, str) or not isinstance(area, str):
        return args
    needle = name.casefold().strip()
    matches = [e for e in home.entities if needle in labels(e)]
    if len(matches) != 1:
        return args
    entity = matches[0]
    if entity.area is not None and area.casefold() == entity.area.casefold():
        return args
    ut = tokens(utterance)
    names = [area.casefold(), *(a.casefold() for a in home.areas.get(area, ()))]
    if any(_phrase_in_utterance(w, ut) for w in names):
        return args
    return {k: v for k, v in args.items() if k != "area"}


def call_domain(tool_name: str, args: dict) -> str | None:
    prefix = tool_name.split("__")[0] if "__" in tool_name else "intent"
    if prefix != "intent":
        return prefix
    domain = args.get("domain")
    if isinstance(domain, list):
        if len(domain) == 1 and isinstance(domain[0], str):
            return domain[0]
        return None
    if isinstance(domain, str):
        return domain
    return None


def _matches_dom(entity: GroundingEntity, dom: str | None) -> bool:
    if dom is None or entity.domain == dom:
        return True
    name = entity.name.casefold()
    return (
        dom == "light"
        and entity.domain == "switch"
        and ("lamp" in name or "light" in name)
    )


def _pool(utterance: str, home: GroundingHome, dom: str | None, room: str | None):
    pool = [e for e in home.entities if _matches_dom(e, dom)]
    if home.satellite_device_id is not None:
        pool = [e for e in pool if e.device_id != home.satellite_device_id]
    if room is not None:
        room_cf = room.casefold()
        pool = [e for e in pool if (e.area or "").casefold() == room_cf]
    return pool


def _phrase_score(label: str, utterance: str) -> float:
    ut = tokens(utterance)
    d = " ".join(t for t in tokens(label) if t not in GENERIC)
    if not d:
        return 0.0
    n = len(d.split())
    best = 0.0
    for i, tok in enumerate(ut):
        if tok not in NOUN:
            continue
        for k in (n, n + 1):
            if i - k < 0:
                continue
            w = " ".join(ut[i - k : i])
            if not w[:1] or w[:1] != d[:1]:
                continue
            r = word_ratio(d, w)
            best = max(best, r)
    return best


def _score(entity: GroundingEntity, utterance: str, home: GroundingHome, room):
    base = 0.0
    for l in labels(entity):
        s = max(support(l, utterance, generic_only=0.3), _phrase_score(l, utterance))
        base = max(base, s)
    if (
        room is None
        and home.satellite_area is not None
        and entity.area == home.satellite_area
    ):
        base += 0.2
    return base


def _area_noun_match(
    entity: GroundingEntity, utterance: str, home: GroundingHome
) -> bool:
    if entity.area is None:
        return False
    room = mentioned_area(utterance, home)
    if room is None or room.casefold() != entity.area.casefold():
        return False
    nouns = [NOUN[t] for t in tokens(utterance) if t in NOUN]
    name_cf = entity.name.casefold()
    for nd in nouns:
        if nd == entity.domain:
            return True
        if (
            nd == "light"
            and entity.domain == "switch"
            and ("lamp" in name_cf or "light" in name_cf)
        ):
            return True
    return False


def is_grounded(
    tool_name: str, args: dict, utterance: str, home: GroundingHome
) -> bool:
    bare = tool_name.split("__")[-1]
    if bare in _ALWAYS_GROUNDED or "Timer" in bare:
        return True
    name = args.get("name")
    if isinstance(name, str):
        needle = name.casefold().strip()
        matches = [e for e in home.entities if needle in labels(e)]
        if matches:
            first = matches[0]
            if max(support(l, utterance) for l in labels(first)) >= 0.5:
                return True
            return _area_noun_match(first, utterance, home)
        for area_name, aliases in home.areas.items():
            names = {
                area_name.casefold().strip(),
                *(a.casefold().strip() for a in aliases),
            }
            if needle in names:
                return _area_uttered(area_name, utterance, home)
        return False
    area = args.get("area")
    if isinstance(area, str):
        return _area_uttered(area, utterance, home)
    return True


def clarify(
    tool_name: str, args: dict, utterance: str, home: GroundingHome
) -> Clarification:
    nouns = [NOUN[t] for t in tokens(utterance) if t in NOUN]
    dom = nouns[-1] if nouns else call_domain(tool_name, args)
    room = mentioned_area(utterance, home)
    pool = _pool(utterance, home, dom, room)
    if not pool:
        if dom is None:
            plural = "devices"
        else:
            plural = DOMAIN_PLURALS.get(dom, f"{dom.replace('_', ' ')}s")
        speech = ERROR_AREA_HAS_NO_DOMAIN.format(
            area=room.casefold() if room else "home", domains=plural
        )
        return Clarification(speech=speech, options=())
    pool.sort(key=lambda e: e.name.casefold())
    scored = [(_score(e, utterance, home, room), e) for e in pool]
    scored.sort(key=lambda se: -se[0])
    if len(scored) == 1:
        return Clarification(
            speech=ERROR_DID_YOU_MEAN.format(name=scored[0][1].name),
            options=(scored[0][1],),
        )
    s1, e1 = scored[0]
    s2, e2 = scored[1]
    if s1 >= 0.5 and s1 - s2 >= 0.1:
        return Clarification(
            speech=ERROR_DID_YOU_MEAN.format(name=e1.name), options=(e1,)
        )
    if s1 >= 0.5 and s2 >= 0.5:
        speech = ERROR_CHOOSE.format(first=e1.name, second=e2.name)
        return Clarification(speech=speech, options=(e1, e2))
    return Clarification(speech=ERROR_NOT_UNDERSTOOD, options=())
