import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.wake_tts_distant import near_koda


def test_near_koda_catches_neighbours_and_passes_distant_speech():
    assert near_koda("K OW D AH".split())  # Koda
    assert near_koda("K OW D".split())  # code
    assert near_koda("S OW D AH".split())  # soda
    assert near_koda("DH AH K OW L D".split())  # the cold
    assert not near_koda("M EY B IY IH T".split())  # maybe it
    assert not near_koda("F AE SH AH N".split())  # fashion
