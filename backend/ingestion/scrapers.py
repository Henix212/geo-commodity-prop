"""Free news / RSS scrapers for commodity shock signals.

Sources:
  - RSS: Mining.com, OilPrice, EIA, Google News (configurable via GCP_RSS_FEEDS)
  - GDELT DOC 2.0 API (no API key)
  - NASA EONET natural events (no API key)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus, urlparse

import requests

from backend import config

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 30


@dataclass
class Article:
    source: str
    title: str
    url: str
    published: str | None = None
    summary: str = ""
    fetched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def uid(self) -> str:
        basis = (self.url or self.title).strip().lower()
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "*/*"})
    return s


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError, IndexError, OverflowError):
        return value.strip() or None


def _source_name(url: str, fallback: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or fallback


def _local(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _child_text(el: ET.Element, *names: str) -> str | None:
    wanted = set(names)
    for child in list(el):
        if _local(child.tag) in wanted:
            if child.text and child.text.strip():
                return child.text.strip()
            # some feeds put title in nested elements
            joined = "".join(child.itertext()).strip()
            return joined or None
    return None


def _child_link(el: ET.Element) -> str | None:
    for child in list(el):
        if _local(child.tag) != "link":
            continue
        href = child.attrib.get("href")
        if href:
            return href.strip()
        if child.text and child.text.strip():
            return child.text.strip()
    return None


def parse_rss_xml(xml_text: str, feed_url: str) -> list[Article]:
    """Parse RSS 2.0 or Atom XML into Article list."""
    root = ET.fromstring(xml_text)
    source = _source_name(feed_url, "rss")
    articles: list[Article] = []

    # RSS 2.0
    channel = root.find("channel")
    if channel is not None:
        for item in channel.findall("item"):
            title = _clean_text(_child_text(item, "title") or "")
            link = _child_text(item, "link") or _child_link(item) or ""
            summary = _clean_text(
                _child_text(item, "description", "summary", "content") or ""
            )
            published = _parse_date(
                _child_text(item, "pubDate", "published", "updated")
            )
            if not title and not link:
                continue
            articles.append(
                Article(
                    source=source,
                    title=title or link,
                    url=link,
                    published=published,
                    summary=summary,
                )
            )
        return articles

    # Atom
    entries = [el for el in root.iter() if _local(el.tag) == "entry"]
    for entry in entries:
        title = _clean_text(_child_text(entry, "title") or "")
        link = _child_link(entry) or _child_text(entry, "id") or ""
        summary = _clean_text(
            _child_text(entry, "summary", "content", "description") or ""
        )
        published = _parse_date(_child_text(entry, "published", "updated"))
        if not title and not link:
            continue
        articles.append(
            Article(
                source=source,
                title=title or link,
                url=link,
                published=published,
                summary=summary,
            )
        )
    return articles


def fetch_rss(
    feed_url: str,
    *,
    session: requests.Session | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[Article]:
    sess = session or _session()
    logger.info("Fetching RSS %s", feed_url)
    resp = sess.get(feed_url, timeout=timeout)
    resp.raise_for_status()
    return parse_rss_xml(resp.text, feed_url)


def fetch_eonet(
    *,
    days: int = 30,
    session: requests.Session | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[Article]:
    """NASA EONET natural-event feed (free, no key) — weather / disaster shocks."""
    sess = session or _session()
    url = f"https://eonet.gsfc.nasa.gov/api/v3/events?days={days}&status=open"
    logger.info("Fetching NASA EONET days=%s", days)
    resp = sess.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    articles: list[Article] = []
    for event in payload.get("events") or []:
        title = _clean_text(event.get("title") or "")
        link = ""
        for src in event.get("sources") or []:
            if src.get("url"):
                link = src["url"]
                break
        cats = ", ".join(c.get("title", "") for c in (event.get("categories") or []))
        geometry = event.get("geometry") or []
        published = None
        if geometry:
            published = _parse_date(geometry[-1].get("date")) or geometry[-1].get("date")
        if not title:
            continue
        articles.append(
            Article(
                source="nasa:eonet",
                title=title,
                url=link or f"https://eonet.gsfc.nasa.gov/api/v3/events/{event.get('id', '')}",
                published=published,
                summary=_clean_text(cats),
            )
        )
    return articles


def fetch_gdelt(
    query: str | None = None,
    *,
    timespan: str = "24h",
    maxrecords: int = 75,
    session: requests.Session | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> list[Article]:
    """Fetch recent articles from GDELT DOC 2.0 (no API key)."""
    sess = session or _session()
    q = query or config.GDELT_QUERY
    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={quote_plus(q)}"
        f"&mode=ArtList&format=json"
        f"&maxrecords={maxrecords}"
        f"&timespan={timespan}"
        f"&sort=DateDesc"
    )
    logger.info("Fetching GDELT timespan=%s maxrecords=%s", timespan, maxrecords)
    resp = sess.get(url, timeout=timeout)
    resp.raise_for_status()
    # GDELT sometimes returns empty body or HTML on overload
    text = resp.text.strip()
    if not text or text.startswith("<"):
        logger.warning("GDELT returned non-JSON body (len=%d)", len(text))
        return []
    payload = resp.json()
    articles: list[Article] = []
    for item in payload.get("articles") or []:
        title = _clean_text(item.get("title") or "")
        link = (item.get("url") or "").strip()
        if not title and not link:
            continue
        seendate = item.get("seendate")
        published = None
        if seendate and len(seendate) >= 14:
            # YYYYMMDDHHMMSS
            try:
                published = datetime.strptime(seendate[:14], "%Y%m%d%H%M%S").replace(
                    tzinfo=timezone.utc
                ).isoformat()
            except ValueError:
                published = seendate
        domain = item.get("domain") or _source_name(link, "gdelt")
        articles.append(
            Article(
                source=f"gdelt:{domain}",
                title=title or link,
                url=link,
                published=published,
                summary=_clean_text(
                    " ".join(
                        filter(
                            None,
                            [
                                item.get("sourcecountry"),
                                item.get("language"),
                                item.get("socialimage"),
                            ],
                        )
                    )
                ),
            )
        )
    return articles


def dedupe_articles(articles: Iterable[Article]) -> list[Article]:
    seen: set[str] = set()
    out: list[Article] = []
    for art in articles:
        key = art.url.strip().lower() if art.url else art.uid
        if key in seen:
            continue
        seen.add(key)
        out.append(art)
    return out


def scrape_all(
    *,
    rss_feeds: list[str] | None = None,
    gdelt_query: str | None = None,
    gdelt_timespan: str = "24h",
    gdelt_maxrecords: int = 75,
    include_gdelt: bool = True,
    include_eonet: bool = True,
    sleep_s: float = 0.4,
) -> list[Article]:
    """Scrape configured RSS feeds + optional GDELT + NASA EONET."""
    feeds = rss_feeds if rss_feeds is not None else list(config.RSS_FEEDS)
    sess = _session()
    collected: list[Article] = []

    for feed in feeds:
        try:
            collected.extend(fetch_rss(feed, session=sess))
        except Exception as exc:  # noqa: BLE001 — keep scraping other feeds
            logger.warning("RSS failed %s: %s", feed, exc)
        time.sleep(sleep_s)

    if include_gdelt:
        try:
            collected.extend(
                fetch_gdelt(
                    gdelt_query,
                    timespan=gdelt_timespan,
                    maxrecords=gdelt_maxrecords,
                    session=sess,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("GDELT failed: %s", exc)

    if include_eonet:
        try:
            collected.extend(fetch_eonet(session=sess))
        except Exception as exc:  # noqa: BLE001
            logger.warning("EONET failed: %s", exc)

    return dedupe_articles(collected)


def save_articles(
    articles: list[Article],
    path: Path | None = None,
    *,
    persist_db: bool = True,
) -> Path:
    """Persist articles to SQLite (upsert) and write latest batch to JSONL."""
    from backend.database.models import upsert_articles

    config.ensure_dirs()

    if persist_db:
        result = upsert_articles(articles)
        logger.info(
            "DB upsert: +%d new, ~%d updated, %d total in %s",
            result.inserted,
            result.updated,
            result.total,
            config.DB_PATH,
        )

    out = path or (config.DATA_DIR / "raw_articles.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for art in articles:
            row = asdict(art)
            row["uid"] = art.uid
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d articles (latest batch) → %s", len(articles), out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scrape free commodity news sources")
    parser.add_argument("--no-gdelt", action="store_true", help="Skip GDELT API")
    parser.add_argument("--no-eonet", action="store_true", help="Skip NASA EONET")
    parser.add_argument("--timespan", default="24h", help="GDELT timespan (e.g. 24h, 7d)")
    parser.add_argument("--maxrecords", type=int, default=75, help="GDELT max records")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path (default: data/raw_articles.jsonl)",
    )
    parser.add_argument("--log-level", default=config.LOG_LEVEL)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    articles = scrape_all(
        include_gdelt=not args.no_gdelt,
        include_eonet=not args.no_eonet,
        gdelt_timespan=args.timespan,
        gdelt_maxrecords=args.maxrecords,
    )
    path = save_articles(articles, args.out)
    print(f"scraped={len(articles)} out={path}")
    for art in articles[:5]:
        print(f"- [{art.source}] {art.title[:100]}")


if __name__ == "__main__":
    main()
