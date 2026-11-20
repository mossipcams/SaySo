from __future__ import annotations

import difflib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .vocabulary import SPEECH_EN, VOCAB_EN, Speech, Vocabulary

_TOKEN_RE = re.compile(r"[a-z0-9']+")

_ALWAYS_GROUNDED = {"GetLiveContext", "GetDateTime"}

PARTIAL_FLOOR = 1 / 3


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


def support(
    label: str,
    utterance: str,
    generic_only: float = 1.0,
    *,
    vocab: Vocabulary = VOCAB_EN,
) -> float:
    ut = tokens(utterance)
    d = [t for t in tokens(label) if t not in vocab.generic]
    u = [t for t in ut if t not in vocab.generic and t not in vocab.noun] or [""]
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
            for i in range(len(ut)):
                if ut[i : i + n] == ws and (best is None or n > best[0]):
                    best = (n, area)
    return best[1] if best else None


def _phrase_in_utterance(phrase: str, ut_tokens: list[str]) -> bool:
    ws = [p for p in re.split(r"[^a-z0-9']+", phrase.casefold()) if p]
    if not ws:
        return False
    n = len(ws)
    return any(ut_tokens[i : i + n] == ws for i in range(len(ut_tokens) - n + 1))


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
    tool_name: str, args: dict[str, Any], utterance: str, home: GroundingHome
) -> dict[str, Any]:
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


def call_domain(tool_name: str, args: dict[str, Any]) -> str | None:
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


def _pool(
    utterance: str, home: GroundingHome, dom: str | None, room: str | None
) -> list[GroundingEntity]:
    pool = [e for e in home.entities if _matches_dom(e, dom)]
    if home.satellite_device_id is not None:
        pool = [e for e in pool if e.device_id != home.satellite_device_id]
    if room is not None:
        room_cf = room.casefold()
        pool = [e for e in pool if (e.area or "").casefold() == room_cf]
    return pool


def _phrase_score(label: str, utterance: str, vocab: Vocabulary) -> float:
    ut = tokens(utterance)
    d = " ".join(t for t in tokens(label) if t not in vocab.generic)
    if not d:
        return 0.0
    n = len(d.split())
    best = 0.0
    for i, tok in enumerate(ut):
        if tok not in vocab.noun:
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


def _score(
    entity: GroundingEntity,
    utterance: str,
    home: GroundingHome,
    room: str | None,
    vocab: Vocabulary,
) -> float:
    base = 0.0
    for l in labels(entity):
        s = max(
            support(l, utterance, generic_only=0.3, vocab=vocab),
            _phrase_score(l, utterance, vocab),
        )
        base = max(base, s)
    if (
        room is None
        and home.satellite_area is not None
        and entity.area == home.satellite_area
    ):
        base += 0.2
    return base


def _area_noun_match(
    entity: GroundingEntity,
    utterance: str,
    home: GroundingHome,
    vocab: Vocabulary = VOCAB_EN,
) -> bool:
    if entity.area is None:
        return False
    room = mentioned_area(utterance, home)
    if room is None or room.casefold() != entity.area.casefold():
        return False
    nouns = [vocab.noun[t] for t in tokens(utterance) if t in vocab.noun]
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


def _same_domain_word(word: str, other: str) -> bool:
    return word.removesuffix("s") == other.removesuffix("s")


def _room_ok(
    entity: GroundingEntity,
    utterance: str,
    home: GroundingHome,
    bare: str,
    args: dict[str, Any],
) -> bool:
    room = mentioned_area(utterance, home)
    if room is None:
        return True
    area = args.get("area")
    if (
        "CleanArea" in bare
        and isinstance(area, str)
        and area.casefold() == room.casefold()
    ):
        return True
    return entity.area is not None and entity.area.casefold() == room.casefold()


def _area_tokens(area: str | None, home: GroundingHome) -> set[str]:
    if area is None:
        return set()
    names = [area, *home.areas.get(area, ())]
    return {t for name in names for t in tokens(name)}


def _batch_words(
    others: Sequence[tuple[str, dict[str, Any]]], home: GroundingHome
) -> set[str]:
    words: set[str] = set()
    for tool_name, args in others:
        bare = tool_name.split("__")[-1]
        name = args.get("name")
        if isinstance(name, str):
            needle = name.casefold().strip()
            matches = [e for e in home.entities if needle in labels(e)]
            if matches:
                target = matches[0]
                words |= {t for label in labels(target) for t in tokens(label)}
                words |= _area_tokens(target.area, home)
        if "CleanArea" in bare:
            area = args.get("area")
            if isinstance(area, str):
                words |= set(tokens(area))
    return words


def _unexplained(
    entity: GroundingEntity,
    utterance: str,
    home: GroundingHome,
    bare: str,
    args: dict[str, Any],
    others: Sequence[tuple[str, dict[str, Any]]],
    vocab: Vocabulary,
) -> list[str]:
    words = set(vocab.stop_words) | set(vocab.domain_words)
    words |= {t for label in labels(entity) for t in tokens(label)}
    words |= _area_tokens(entity.area, home)
    words |= _area_tokens(home.satellite_area, home)
    if "CleanArea" in bare:
        area = args.get("area")
        if isinstance(area, str):
            words |= set(tokens(area))
    words |= _batch_words(others, home)
    return [t for t in tokens(utterance) if t not in words and not t.isdigit()]


def _other_candidate_word(
    entity: GroundingEntity,
    utterance: str,
    pool: list[GroundingEntity],
    vocab: Vocabulary,
) -> bool:
    own = {t for label in labels(entity) for t in tokens(label)}
    other = {
        t for e in pool if e != entity for label in labels(e) for t in tokens(label)
    }
    return any(
        t in other and t not in own and t not in vocab.stop_words
        for t in tokens(utterance)
    )


def _apostrophe_unaccounted(
    utterance: str, home: GroundingHome, others: Sequence[tuple[str, dict[str, Any]]]
) -> bool:
    if not others:
        return "'" in utterance
    batch = _batch_words(others, home)
    return any("'" in t and t not in batch for t in tokens(utterance))


def _in_satellite_area(entity: GroundingEntity, home: GroundingHome) -> bool:
    sat = home.satellite_area
    return (
        sat is not None
        and entity.area is not None
        and entity.area.casefold() == sat.casefold()
    )


def _satellite_unique(
    entity: GroundingEntity,
    utterance: str,
    home: GroundingHome,
    pool: list[GroundingEntity],
    vocab: Vocabulary,
) -> bool:
    sat = home.satellite_area
    local = [
        e
        for e in pool
        if e.area is not None
        and sat is not None
        and e.area.casefold() == sat.casefold()
    ]
    if local == [entity]:
        return True
    spoken = [t for t in tokens(utterance) if t in vocab.domain_words]
    narrowed = [
        e
        for e in local
        if any(_same_domain_word(w, t) for w in spoken for t in tokens(e.name))
    ]
    return narrowed == [entity]


def _satellite_local(
    entity: GroundingEntity,
    utterance: str,
    home: GroundingHome,
    pool: list[GroundingEntity],
    others: Sequence[tuple[str, dict[str, Any]]],
    vocab: Vocabulary,
) -> bool:
    if not _in_satellite_area(entity, home):
        return False
    if _apostrophe_unaccounted(utterance, home, others) or _other_candidate_word(
        entity, utterance, pool, vocab
    ):
        return False
    return _satellite_unique(entity, utterance, home, pool, vocab)


def _misheard_room_word(
    entity: GroundingEntity,
    utterance: str,
    home: GroundingHome,
    pool: list[GroundingEntity],
    others: Sequence[tuple[str, dict[str, Any]]],
    dom: str,
    unexplained: list[str],
    vocab: Vocabulary,
) -> bool:
    if len(unexplained) != 1 or not _in_satellite_area(entity, home):
        return False
    word = unexplained[0]
    ut = tokens(utterance)
    before_domain = any(
        ut[i] == word and i + 1 < len(ut) and vocab.domain_words.get(ut[i + 1]) == dom
        for i in range(len(ut))
    )
    if not before_domain or word in vocab.place_words:
        return False
    if any(word in tokens(label) for e in pool if e != entity for label in labels(e)):
        return False
    if _apostrophe_unaccounted(utterance, home, others):
        return False
    return _satellite_unique(entity, utterance, home, pool, vocab)


def _partial_name(
    entity: GroundingEntity,
    utterance: str,
    pool: list[GroundingEntity],
    vocab: Vocabulary,
) -> bool:
    if entity not in pool:
        return False
    if max(support(l, utterance, vocab=vocab) for l in labels(entity)) < PARTIAL_FLOOR:
        return False
    return all(
        max(support(l, utterance, vocab=vocab) for l in labels(other)) < PARTIAL_FLOOR
        for other in pool
        if other != entity
    )


def _rule_grounded(
    entity: GroundingEntity,
    utterance: str,
    home: GroundingHome,
    bare: str,
    args: dict[str, Any],
    others: Sequence[tuple[str, dict[str, Any]]],
    vocab: Vocabulary,
) -> bool:
    if not _room_ok(entity, utterance, home, bare, args):
        return False
    unexplained = _unexplained(entity, utterance, home, bare, args, others, vocab)
    for dom in {
        vocab.domain_words[t] for t in tokens(utterance) if t in vocab.domain_words
    }:
        if not _matches_dom(entity, dom):
            continue
        pool = _pool(utterance, home, dom, None)
        if unexplained:
            if _misheard_room_word(
                entity, utterance, home, pool, others, dom, unexplained, vocab
            ):
                return True
            continue
        if pool == [entity]:
            return True
        if _satellite_local(entity, utterance, home, pool, others, vocab):
            return True
        if _partial_name(entity, utterance, pool, vocab):
            return True
    return False


def is_grounded(
    tool_name: str,
    args: dict[str, Any],
    utterance: str,
    home: GroundingHome,
    *,
    others: Sequence[tuple[str, dict[str, Any]]] = (),
    vocab: Vocabulary = VOCAB_EN,
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
            if max(support(l, utterance, vocab=vocab) for l in labels(first)) >= 0.5:
                return True
            if _area_noun_match(first, utterance, home, vocab):
                return True
            return _rule_grounded(first, utterance, home, bare, args, others, vocab)
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


def _satellite_candidate(
    utterance: str,
    home: GroundingHome,
    room: str | None,
    vocab: Vocabulary,
) -> GroundingEntity | None:
    sat = home.satellite_area
    if sat is None or (room is not None and room.casefold() != sat.casefold()):
        return None
    spoken = {
        vocab.domain_words[t] for t in tokens(utterance) if t in vocab.domain_words
    }
    if len(spoken) != 1:
        return None
    (domain,) = spoken
    local = [
        e
        for e in _pool(utterance, home, domain, None)
        if e.area is not None and e.area.casefold() == sat.casefold()
    ]
    return local[0] if len(local) == 1 else None


def clarify(
    tool_name: str,
    args: dict[str, Any],
    utterance: str,
    home: GroundingHome,
    *,
    speech: Speech = SPEECH_EN,
    vocab: Vocabulary = VOCAB_EN,
) -> Clarification:
    nouns = [vocab.noun[t] for t in tokens(utterance) if t in vocab.noun]
    dom = nouns[-1] if nouns else call_domain(tool_name, args)
    room = mentioned_area(utterance, home)
    pool = _pool(utterance, home, dom, room)
    if not pool:
        if dom is None:
            plural = "devices"
        else:
            plural = vocab.domain_plurals.get(dom, f"{dom.replace('_', ' ')}s")
        return Clarification(
            speech=speech.area_has_no_domain.format(
                area=room.casefold() if room else "home", domains=plural
            ),
            options=(),
        )
    pool.sort(key=lambda e: e.name.casefold())
    scored = [(_score(e, utterance, home, room, vocab), e) for e in pool]
    scored.sort(key=lambda se: -se[0])
    if len(scored) == 1:
        return Clarification(
            speech=speech.did_you_mean.format(name=scored[0][1].name),
            options=(scored[0][1],),
        )
    s1, e1 = scored[0]
    s2, e2 = scored[1]
    if s1 >= 0.5 and s1 - s2 >= 0.1:
        return Clarification(
            speech=speech.did_you_mean.format(name=e1.name), options=(e1,)
        )
    if s1 >= 0.5 and s2 >= 0.5:
        return Clarification(
            speech=speech.choose.format(first=e1.name, second=e2.name),
            options=(e1, e2),
        )
    candidate = _satellite_candidate(utterance, home, room, vocab)
    if candidate is not None:
        return Clarification(
            speech=speech.did_you_mean.format(name=candidate.name),
            options=(candidate,),
        )
    return Clarification(speech=speech.not_understood, options=())
