"""Resolve semantic include/exclude names within a scope entity set."""

from __future__ import annotations

from sayso_server.home_graph import HomeGraphSnapshot
from sayso_server.normalize import normalize_labels, normalize_tokens
from sayso_server.scoring import CandidateItem, content_tokens


def resolve_names_in_scope(
    snapshot: HomeGraphSnapshot,
    scope_entity_ids: frozenset[str],
    names: list[str],
) -> frozenset[str]:
    """Return entity IDs in scope whose names or aliases match any semantic name."""
    if not names or not scope_entity_ids:
        return frozenset()

    items_by_id = _items_by_entity_id(snapshot)
    matched: set[str] = set()
    for name in names:
        for entity_id in scope_entity_ids:
            item = items_by_id.get(entity_id)
            if item is not None and matches_semantic_name(name, item):
                matched.add(entity_id)
    return frozenset(sorted(matched))


def apply_inclusions_exclusions(
    snapshot: HomeGraphSnapshot,
    base_entity_ids: frozenset[str],
    *,
    targets: list[str] | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> frozenset[str]:
    """Resolve names inside base scope and subtract exclusions."""
    names = [*(targets or []), *(include or [])]
    if names:
        result = resolve_names_in_scope(snapshot, base_entity_ids, names)
    else:
        result = base_entity_ids

    if exclude:
        excluded = resolve_names_in_scope(snapshot, base_entity_ids, exclude)
        result = result - excluded

    return frozenset(sorted(result))


def matches_semantic_name(name: str, item: CandidateItem) -> bool:
    """Return whether a semantic name matches an entity, scene, or script label."""
    labels = [item.name, *item.aliases]
    label_tokens = normalize_labels(labels)
    name_tokens = content_tokens(normalize_tokens(name))
    token_set = set(name_tokens)
    if not token_set:
        return False

    if token_set.intersection(label_tokens):
        return True

    for alias_tokens in (normalize_tokens(label) for label in labels):
        if alias_tokens and set(alias_tokens).issubset(token_set):
            return True

    joined_query = " ".join(name_tokens)
    for label in labels:
        normalized_label = " ".join(normalize_tokens(label))
        if normalized_label and normalized_label in joined_query:
            return True

    return False


def domain_matches(item_domain: str, requested_domain: str) -> bool:
    """Return whether an item domain satisfies a requested control domain."""
    if requested_domain == "light":
        return item_domain in ("light", "switch")
    return item_domain == requested_domain


def filter_entity_ids_by_domain(
    snapshot: HomeGraphSnapshot,
    entity_ids: frozenset[str],
    domain: str,
) -> frozenset[str]:
    """Keep only entities, scenes, or scripts matching the requested domain."""
    items_by_id = _items_by_entity_id(snapshot)
    matched = sorted(
        entity_id
        for entity_id in entity_ids
        if (item := items_by_id.get(entity_id)) is not None
        and domain_matches(_item_domain(item), domain)
    )
    return frozenset(matched)


def _items_by_entity_id(snapshot: HomeGraphSnapshot) -> dict[str, CandidateItem]:
    return {
        item.entity_id: item
        for item in (*snapshot.entities, *snapshot.scenes, *snapshot.scripts)
    }


def _item_domain(item: CandidateItem) -> str:
    from sayso_server.home_graph import Entity, Scene

    if isinstance(item, Entity):
        return item.domain
    if isinstance(item, Scene):
        return "scene"
    return "script"
