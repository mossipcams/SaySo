"""Smoke test that the sayso_satellite package imports after install."""


def test_import_sayso_satellite() -> None:
    import sayso_satellite

    assert sayso_satellite.__version__
