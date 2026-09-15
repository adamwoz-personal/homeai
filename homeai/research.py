# Copyright (c) 2026 Adam Wosotowsky
# SPDX-License-Identifier: BUSL-1.1
# Licensed under the Business Source License 1.1; see LICENSE at the
# repository root. Converts to Apache-2.0 on 2030-09-15.
"""Web research: search, fetch, extract, and hand back sourced excerpts.

Why this exists
---------------
Asked "what do you think about modified gravity versus string theory", the
assistant searched for the bare token "MONDS" and returned a Rainmeter theme
and two American football recruits. Three separate failures compounded:

1. **No conversational context**, so the follow-up was searched cold.
2. **A search tool that returns titles and snippets only.** Snippets are
   advertising copy. You cannot form an opinion from them, so the model either
   parrots them or invents something.
3. **No signal about what is worth reading**, so an obviously irrelevant
   result was treated as authoritative.

This module addresses (2) and (3): it fetches the actual pages, extracts
readable text, and returns substantial sourced excerpts the model can reason
over. Context (1) is handled separately in the agent client.

Design notes
------------
*Sources are always returned with the text.* A research answer that cannot be
traced back to a page is indistinguishable from a hallucination, and the
weather bug showed how convincing a hallucination sounds.

*Failure is explicit.* If nothing can be fetched, this says so rather than
returning an empty string that the model would happily fill with invention.

No API keys: DuckDuckGo's HTML endpoint needs no account, which keeps this
working on someone else's machine with no setup. It is also rate-limited and
occasionally changes markup, so parsing degrades to "no results" rather than
raising.
"""

from __future__ import annotations

import concurrent.futures
import html
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger("homeai.research")

_SEARCH_URL = "https://html.duckduckgo.com/html/"

# A browser-shaped UA. Several sites return a consent wall or an empty body to
# obviously-automated clients, which previously looked like "no results".
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_RESULT_LINK = re.compile(
    r'<a[^>]+class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)

_SCRIPTISH = re.compile(
    r"<(script|style|noscript|svg|nav|header|footer|form)\b.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n{3,}")

# Domains that are never worth fetching for a factual question: they are
# either aggregators of other pages or content this assistant should not cite.
_SKIP_DOMAINS = (
    "pinterest.",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "x.com",
    "twitter.com",
)


@dataclass
class Source:
    title: str
    url: str
    text: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text) and not self.error

    @property
    def domain(self) -> str:
        try:
            return urllib.parse.urlparse(self.url).netloc
        except ValueError:
            return ""


@dataclass
class Research:
    query: str
    sources: list[Source] = field(default_factory=list)
    error: str = ""

    @property
    def usable(self) -> list[Source]:
        return [s for s in self.sources if s.ok]

    def as_prompt(self, per_source_chars: int = 2000) -> str:
        """Format for consumption by the model.

        Sources are numbered and labelled with their domain so the model can
        weigh them -- a university page and a forum post should not carry
        equal weight, and it can only make that judgement if it can see where
        each excerpt came from.
        """
        if self.error:
            return f"Research failed: {self.error}"
        usable = self.usable
        if not usable:
            return (
                f"No readable sources were found for '{self.query}'. Say that "
                "you could not find reliable information rather than guessing."
            )

        chunks = [f"Research results for '{self.query}':", ""]
        for index, source in enumerate(usable, start=1):
            chunks.append(f"[{index}] {source.title} ({source.domain})")
            chunks.append(source.text[:per_source_chars].strip())
            chunks.append("")

        chunks.append(
            "Using only the sources above, answer the question and give your "
            "own assessment. If the sources disagree or are inadequate, say "
            "so. Do not state anything the sources do not support."
        )
        return "\n".join(chunks)


def _fetch(url: str, timeout: float, max_bytes: int = 600_000) -> str:
    """Fetch a URL as text, capped so one huge page cannot exhaust memory."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        ctype = response.headers.get("Content-Type", "")
        # Binary payloads (PDF, images) would otherwise become mojibake that
        # the model treats as content.
        if ctype and not any(t in ctype for t in ("html", "text", "xml", "json")):
            raise ValueError(f"unsupported content type: {ctype}")
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read(max_bytes).decode(charset, "replace")


def extract_text(markup: str) -> str:
    """Reduce HTML to readable prose."""
    without_scripts = _SCRIPTISH.sub(" ", markup)
    # Preserve block boundaries so sentences do not weld together.
    spaced = re.sub(
        r"</(p|div|li|h[1-6]|tr|section|article|br)\s*>", "\n", without_scripts,
        flags=re.IGNORECASE,
    )
    text = html.unescape(_TAG.sub(" ", spaced))
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANKS.sub("\n\n", text).strip()


def search(query: str, limit: int = 5, timeout: float = 12.0) -> list[Source]:
    """Return search hits. Never raises; an outage yields an empty list."""
    if not query or not query.strip():
        return []

    data = urllib.parse.urlencode({"q": query.strip()}).encode()
    request = urllib.request.Request(
        _SEARCH_URL, data=data, headers={"User-Agent": _USER_AGENT}
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        log.warning("search failed for %r: %s", query, exc)
        return []

    sources: list[Source] = []
    seen: set[str] = set()

    for href, raw_title in _RESULT_LINK.findall(body):
        url = _clean_result_url(href)
        if not url or url in seen:
            continue
        if any(domain in url for domain in _SKIP_DOMAINS):
            continue
        seen.add(url)
        title = html.unescape(_TAG.sub("", raw_title)).strip()
        sources.append(Source(title=title or url, url=url))
        if len(sources) >= limit:
            break

    return sources


def _clean_result_url(href: str) -> str:
    """Unwrap DuckDuckGo's redirector to the real destination."""
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg")
        if target:
            href = target[0]
    return href if href.startswith("http") else ""


def gather(
    query: str,
    limit: int = 4,
    timeout: float = 12.0,
    min_chars: int = 400,
) -> Research:
    """Search, then fetch the hits concurrently.

    Fetching is parallel because it is entirely network-bound; serially this
    took long enough that the assistant felt hung. Individual failures are
    recorded on the source rather than aborting the batch -- one dead link
    should not cost the other three.
    """
    hits = search(query, limit=limit, timeout=timeout)
    if not hits:
        return Research(query=query, error="no search results")

    def _load(source: Source) -> Source:
        try:
            source.text = extract_text(_fetch(source.url, timeout))
            if len(source.text) < min_chars:
                # Consent walls and JS-only pages yield a few words of
                # boilerplate; treating that as a source invites invention.
                source.error = "too little readable text"
                source.text = ""
        except Exception as exc:  # noqa: BLE001 - one bad link must not abort
            source.error = str(exc)
        return source

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(hits), 6)) as pool:
        sources = list(pool.map(_load, hits))

    research = Research(query=query, sources=sources)
    if not research.usable:
        research.error = "search results could not be read"
    return research


def research_prompt(query: str, limit: int = 4) -> str:
    """Convenience wrapper returning model-ready text."""
    return gather(query, limit=limit).as_prompt()
