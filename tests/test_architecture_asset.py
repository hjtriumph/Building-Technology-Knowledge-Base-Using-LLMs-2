import importlib.util
import warnings
from pathlib import Path

import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]


def _load_make_assets():
    path = ROOT / "docs" / "interview_deck" / "make_assets.py"
    spec = importlib.util.spec_from_file_location("make_assets", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_architecture_full_asset(tmp_path: Path) -> None:
    mod = _load_make_assets()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod.architecture_full(tmp_path)

    glyph_warnings = [str(w.message) for w in caught if "Glyph" in str(w.message)]
    assert not glyph_warnings, glyph_warnings

    png = tmp_path / "architecture_full.png"
    svg = tmp_path / "architecture_full.svg"
    assert png.exists() and png.stat().st_size > 50_000
    assert svg.exists()

    svg_text = svg.read_text(encoding="utf-8")
    for expected in ("离线索引", "在线问答", "评测与校准", "Chroma", "1,410 chunks", "误差分析"):
        assert expected in svg_text, expected
    assert svg_text.count("<text") >= 25
    assert ".notdef" not in svg_text

    img = mpimg.imread(png)
    ratio = img.shape[1] / img.shape[0]
    assert abs(ratio - 16 / 9) < 0.01


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_architecture_full_asset(Path(d))
    print("OK")
