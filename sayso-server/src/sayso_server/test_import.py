"""Smoke test that the sayso_server package imports after install."""


def test_import_sayso_server() -> None:
    import sayso_server

    assert sayso_server.__version__
