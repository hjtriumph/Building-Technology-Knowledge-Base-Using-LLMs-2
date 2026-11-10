from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable


class ResponseCache:
    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key(self, key_parts: dict[str, Any]) -> str:
        raw = json.dumps(key_parts, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_or_call(
        self,
        key_parts: dict[str, Any],
        fn: Callable[[], dict[str, Any]],
        enabled: bool = True,
    ) -> dict[str, Any]:
        if not enabled:
            return fn()

        path = self.dir / f"{self._key(key_parts)}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        result = fn()
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result
