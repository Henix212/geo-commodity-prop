"""LLM event extraction → structured shock features for the graph.

Output schema per event:
{
  "entity_text": "Escondida",
  "event_type": "strike",
  "severity": 0.8,
  "commodity": "copper",
  "direction": "supply_down",
  "confidence": 0.75
}

Post-processing: validate → map entity_text → node_id → filter by thresholds.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend import config
from backend.ingestion.entity_resolver import build_gazetteer, resolve_entity

logger = logging.getLogger(__name__)

EVENT_TYPES = frozenset(
    {
        "strike",
        "accident",
        "sanction",
        "weather",
        "congestion",
        "force_majeure",
        "outage",
        "blockade",
        "policy",
        "other",
    }
)
DIRECTIONS = frozenset(
    {"supply_down", "supply_up", "demand_up", "demand_down", "transit_block", "neutral"}
)
COMMODITIES = frozenset(config.TICKERS.keys())

SYSTEM_PROMPT = """You extract commodity-market shock events from news.
Return ONLY a JSON array (no markdown). Each item must have exactly:
{
  "entity_text": string,   // physical place, company, mine, port, chokepoint, country
  "event_type": one of [strike, accident, sanction, weather, congestion, force_majeure, outage, blockade, policy, other],
  "severity": number 0 to 1,
  "commodity": one of [copper, aluminum, silver, gold, oil, lng, wheat, corn],
  "direction": one of [supply_down, supply_up, demand_up, demand_down, transit_block, neutral],
  "confidence": number 0 to 1
}
If no relevant shock, return []. Prefer concrete facilities (Escondida, Hormuz, Suez) over vague regions.
"""


@dataclass
class ShockEvent:
    entity_text: str
    event_type: str
    severity: float
    commodity: str
    direction: str
    confidence: float
    article_uid: str = ""
    node_id: str | None = None
    match_score: float = 0.0
    extracted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, x))


def _normalize_event_type(value: Any) -> str:
    raw = str(value or "other").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "labour_strike": "strike",
        "labor_strike": "strike",
        "walkout": "strike",
        "explosion": "accident",
        "fire": "accident",
        "embargo": "sanction",
        "tariff": "policy",
        "storm": "weather",
        "flood": "weather",
        "hurricane": "weather",
        "drought": "weather",
        "queue": "congestion",
        "delay": "congestion",
        "closure": "outage",
        "shutdown": "outage",
        "force-majeure": "force_majeure",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in EVENT_TYPES else "other"


def _normalize_direction(value: Any) -> str:
    raw = str(value or "neutral").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "disruption": "supply_down",
        "shortage": "supply_down",
        "cut": "supply_down",
        "increase": "supply_up",
        "boost": "supply_up",
        "block": "transit_block",
        "blocked": "transit_block",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in DIRECTIONS else "neutral"


def _normalize_commodity(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    aliases = {"aluminium": "aluminum", "natgas": "lng", "natural_gas": "lng", "wti": "oil", "brent": "oil"}
    raw = aliases.get(raw, raw)
    return raw if raw in COMMODITIES else None


def validate_raw_event(raw: dict) -> dict | None:
    """Validate/normalize one LLM object; return None if unusable."""
    if not isinstance(raw, dict):
        return None
    entity = str(raw.get("entity_text") or "").strip()
    if len(entity) < 2:
        return None
    commodity = _normalize_commodity(raw.get("commodity"))
    if commodity is None:
        return None
    return {
        "entity_text": entity,
        "event_type": _normalize_event_type(raw.get("event_type")),
        "severity": _clamp01(raw.get("severity"), 0.0),
        "commodity": commodity,
        "direction": _normalize_direction(raw.get("direction")),
        "confidence": _clamp01(raw.get("confidence"), 0.0),
    }


def extract_json_payload(text: str) -> list[dict]:
    """Parse JSON array/object from model output (tolerates markdown fences)."""
    if not text:
        return []
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # try full parse
    for candidate in (cleaned,):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return [data]
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    # bracket slice
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
            if isinstance(data, dict):
                return [data]
        except json.JSONDecodeError:
            pass
    return []


def _article_prompt(title: str, summary: str) -> str:
    body = f"Title: {title}\nSummary: {summary or '(none)'}"
    return (
        f"{SYSTEM_PROMPT}\n\nNews:\n{body}\n\nJSON array:"
    )


class EventParser:
    """Lazy-loaded Qwen (or configured HF model) event extractor."""

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_path = model_path or config.LLM_MODEL_PATH
        self.device = device or config.DEVICE
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = self.model_path
        # Prefer GPU if available even when config says cpu
        if self.device == "cpu" and torch.cuda.is_available():
            self.device = "cuda"
            logger.info("CUDA detected — switching device to cuda")

        logger.info(
            "Loading LLM from %s (device=%s). First load can take several minutes on CPU.",
            path,
            self.device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        load_kwargs: dict = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self.device == "cuda":
            load_kwargs["dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            # float16 on CPU is unsupported for many ops; use float32 but warn
            load_kwargs["dtype"] = torch.float32
            logger.warning(
                "Running Qwen2.5-7B on CPU needs ~30GB RAM and is very slow. "
                "Prefer GCP_DEVICE=cuda or parse fewer articles: "
                "`python -m backend.ingestion.parser_llm --limit 1`"
            )

        self._model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
        if self.device != "cuda":
            self._model.to(self.device)
        self._model.eval()
        logger.info("LLM ready")

    def generate(self, prompt: str) -> str:
        import torch

        self._ensure_model()
        assert self._tokenizer is not None and self._model is not None
        messages = [
            {"role": "system", "content": "You are a precise information extraction engine."},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        inputs = self._tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(gen, skip_special_tokens=True)

    def parse_article(
        self,
        *,
        title: str,
        summary: str = "",
        article_uid: str = "",
        gazetteer: dict[str, str] | None = None,
    ) -> list[ShockEvent]:
        prompt = _article_prompt(title, summary)
        try:
            raw_text = self.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM generate failed for %s: %s", article_uid or title[:40], exc)
            return []

        events: list[ShockEvent] = []
        for raw in extract_json_payload(raw_text):
            norm = validate_raw_event(raw)
            if not norm:
                continue
            node_id = None
            score = 0.0
            if gazetteer is not None:
                node_id, score = resolve_entity(norm["entity_text"], gazetteer)
            events.append(
                ShockEvent(
                    entity_text=norm["entity_text"],
                    event_type=norm["event_type"],
                    severity=norm["severity"],
                    commodity=norm["commodity"],
                    direction=norm["direction"],
                    confidence=norm["confidence"],
                    article_uid=article_uid,
                    node_id=node_id,
                    match_score=score,
                )
            )
        return events


def filter_events(
    events: list[ShockEvent],
    *,
    min_severity: float | None = None,
    min_confidence: float | None = None,
    require_node: bool = True,
) -> list[ShockEvent]:
    min_sev = config.EVENT_SEVERITY_MIN if min_severity is None else min_severity
    min_conf = config.EVENT_CONFIDENCE_MIN if min_confidence is None else min_confidence
    out: list[ShockEvent] = []
    for ev in events:
        if ev.severity < min_sev:
            continue
        if ev.confidence < min_conf:
            continue
        if require_node and not ev.node_id:
            continue
        out.append(ev)
    return out


def parse_event_timestamp(value: str | datetime | None) -> datetime | None:
    """Parse ISO extracted_at (or datetime) to timezone-aware UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def event_age_hours(
    extracted_at: str | datetime | None,
    *,
    as_of: datetime | None = None,
) -> float:
    """Hours since event extraction; 0 if timestamp missing/unparseable."""
    ts = parse_event_timestamp(extracted_at)
    if ts is None:
        return 0.0
    now = as_of or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return max(0.0, (now - ts).total_seconds() / 3600.0)


def decay_factor(
    age_hours: float,
    half_life_hours: float | None = None,
) -> float:
    """Exponential half-life decay: factor = 0.5 ** (age / half_life)."""
    hl = config.EVENT_DECAY_HALF_LIFE_HOURS if half_life_hours is None else half_life_hours
    if hl <= 0:
        return 1.0
    return float(0.5 ** (max(0.0, age_hours) / hl))


def effective_severity(
    severity: float,
    extracted_at: str | datetime | None,
    *,
    as_of: datetime | None = None,
    half_life_hours: float | None = None,
) -> tuple[float, float]:
    """Return (decayed_severity, decay_factor) for an event."""
    age = event_age_hours(extracted_at, as_of=as_of)
    factor = decay_factor(age, half_life_hours)
    return float(severity) * factor, factor


def shock_event_from_row(row: dict[str, Any]) -> ShockEvent:
    """Rebuild ShockEvent from a DB / dict row."""
    return ShockEvent(
        entity_text=str(row["entity_text"]),
        event_type=str(row["event_type"]),
        severity=float(row["severity"]),
        commodity=str(row["commodity"]),
        direction=str(row["direction"]),
        confidence=float(row.get("confidence") or 0.0),
        article_uid=str(row.get("article_uid") or ""),
        node_id=row.get("node_id"),
        match_score=float(row.get("match_score") or 0.0),
        extracted_at=str(row.get("extracted_at") or datetime.now(timezone.utc).isoformat()),
    )


def inject_events_into_network(
    network: dict,
    events: list[ShockEvent] | list[dict[str, Any]],
    *,
    as_of: datetime | None = None,
    half_life_hours: float | None = None,
    min_severity: float | None = None,
) -> int:
    """Write max *decayed* severity onto matching nodes as event_severity.

    Decay: severity_eff = severity * 0.5 ** (age_hours / half_life_hours)
    with half-life from ``GCP_EVENT_DECAY_HALF_LIFE_HOURS`` (default 72h).
    Events below ``min_severity`` after decay are skipped.
    """
    min_sev = config.EVENT_SEVERITY_MIN if min_severity is None else min_severity
    by_id = {n["id"]: n for n in network.get("nodes") or [] if n.get("id")}
    touched = 0
    now = as_of or datetime.now(timezone.utc)

    for raw in events:
        ev = raw if isinstance(raw, ShockEvent) else shock_event_from_row(raw)
        if not ev.node_id or ev.node_id not in by_id:
            continue

        decayed, factor = effective_severity(
            ev.severity,
            ev.extracted_at,
            as_of=now,
            half_life_hours=half_life_hours,
        )
        if decayed < min_sev:
            continue

        signed = decayed if ev.direction != "supply_up" else -decayed
        node = by_id[ev.node_id]
        prev = float(node.get("event_severity") or 0.0)
        # keep strongest absolute (decayed) shock
        if abs(signed) >= abs(prev):
            node["event_severity"] = signed
            node["event_severity_raw"] = ev.severity
            node["event_decay_factor"] = factor
            node["event_type"] = ev.event_type
            node["event_direction"] = ev.direction
            node["event_commodity"] = ev.commodity
            node["event_extracted_at"] = ev.extracted_at
            touched += 1
    return touched


def parse_articles(
    articles: list[dict],
    network: dict,
    *,
    parser: EventParser | None = None,
    limit: int | None = None,
) -> list[ShockEvent]:
    gazetteer = build_gazetteer(network.get("nodes") or [])
    parser = parser or EventParser()
    selected = articles[:limit] if limit is not None else articles
    all_events: list[ShockEvent] = []
    for art in selected:
        title = art.get("title") or ""
        summary = art.get("summary") or ""
        uid = art.get("uid") or ""
        extracted = parser.parse_article(
            title=title, summary=summary, article_uid=uid, gazetteer=gazetteer
        )
        all_events.extend(extracted)
    return filter_events(all_events)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Parse scraped articles into shock events")
    parser.add_argument("--limit", type=int, default=5, help="Max articles to parse")
    parser.add_argument("--sector", default=config.DEFAULT_SECTOR)
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    parser.add_argument(
        "--no-require-node",
        action="store_true",
        help="Keep events even if entity did not map to a node_id",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from backend.database import list_articles, upsert_events
    from backend.graph_core import load_sector

    network = load_sector(args.sector, validate=True)
    articles = list_articles(limit=args.limit)
    if not articles:
        print("No articles in DB — run scrapers first.")
        return

    gazetteer = build_gazetteer(network["nodes"])
    ep = EventParser()
    raw_events: list[ShockEvent] = []
    for art in articles:
        raw_events.extend(
            ep.parse_article(
                title=art["title"],
                summary=art.get("summary") or "",
                article_uid=art["uid"],
                gazetteer=gazetteer,
            )
        )
    events = filter_events(raw_events, require_node=not args.no_require_node)
    result = upsert_events(events)
    injected = inject_events_into_network(network, events)
    print(
        f"parsed={len(raw_events)} kept={len(events)} "
        f"db_inserted={result.inserted} injected_nodes={injected}"
    )
    for ev in events[:10]:
        print(
            f"- {ev.commodity}/{ev.event_type} sev={ev.severity:.2f} "
            f"{ev.entity_text!r} -> {ev.node_id} ({ev.direction})"
        )


if __name__ == "__main__":
    main()
