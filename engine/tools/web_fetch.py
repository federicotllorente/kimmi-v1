"""Fetch public HTTP(S) pages with SSRF guards (no private IPs)."""
from __future__ import annotations

import html
import ipaddress
import re
import socket
from html.parser import HTMLParser
from typing import Any, Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

FETCH_TIMEOUT_SEC = 12
MAX_RESPONSE_BYTES = 200_000
MAX_EXTRACT_CHARS = 3_000
USER_AGENT = "KimmiResearchBot/1.0 (+local-assistant)"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._skip_depth = 0
        self._skip_tags = {
            "script",
            "style",
            "noscript",
            "svg",
            "nav",
            "footer",
            "header",
        }
        self._in_title = False
        self.title_parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        elif self._skip_depth == 0:
            self._parts.append(text)

    def get_text(self) -> str:
        return "\n".join(self._parts)

    def get_title(self) -> str:
        return " ".join(self.title_parts).strip()


def validate_public_url(url: str) -> Tuple[bool, str]:
    """Allow only http(s) URLs that resolve to non-private addresses."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "Invalid URL"

    if parsed.scheme not in ("http", "https"):
        return False, "Only http/https URLs are allowed"

    host = (parsed.hostname or "").lower()
    if not host:
        return False, "URL has no host"

    try:
        infos = socket.getaddrinfo(host, None)
        if not infos:
            return False, f"Could not resolve host '{host}'"
        for info in infos:
            ip_str = info[4][0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
            ):
                return False, "Blocked private or local IP address"
    except socket.gaierror:
        return False, f"Could not resolve host '{host}'"

    return True, ""


def html_to_text(body: str) -> Tuple[str, str]:
    """Return (title, text) from HTML or plain body."""
    lower = body.lower()
    if "<html" in lower or "<body" in lower or "<p" in lower or "<div" in lower:
        parser = _TextExtractor()
        try:
            parser.feed(body)
            title = parser.get_title()
            text = parser.get_text()
        except Exception:
            title = ""
            text = re.sub(r"<[^>]+>", " ", body)
        text = html.unescape(text)
    else:
        title = ""
        text = body

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title.strip(), text.strip()


def fetch_url_text(url: str) -> Dict[str, Any]:
    """Fetch a public URL and extract text.

    Returns {ok, url, final_url?, title?, text?, error?}.
    """
    target = (url or "").strip()
    if not target:
        return {"ok": False, "url": target, "error": "empty_url"}

    ok, reason = validate_public_url(target)
    if not ok:
        return {"ok": False, "url": target, "error": reason}

    req = Request(
        target,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,text/plain,*/*",
        },
    )
    try:
        with urlopen(req, timeout=FETCH_TIMEOUT_SEC) as resp:
            final = resp.geturl()
            ok_final, reason_final = validate_public_url(final)
            if not ok_final:
                return {
                    "ok": False,
                    "url": target,
                    "final_url": final,
                    "error": f"Redirect blocked: {reason_final}",
                }
            raw = resp.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raw = raw[:MAX_RESPONSE_BYTES]
            charset = "utf-8"
            ctype = resp.headers.get_content_charset()
            if ctype:
                charset = ctype
            body = raw.decode(charset, errors="replace")
    except HTTPError as exc:
        return {"ok": False, "url": target, "error": f"HTTP {exc.code}"}
    except URLError as exc:
        return {"ok": False, "url": target, "error": f"unreachable: {exc.reason}"}
    except TimeoutError:
        return {"ok": False, "url": target, "error": "timeout"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": target, "error": str(exc)}

    title, text = html_to_text(body)
    if not text:
        return {
            "ok": False,
            "url": target,
            "final_url": final,
            "title": title,
            "error": "no_extractable_text",
        }

    return {
        "ok": True,
        "url": target,
        "final_url": final,
        "title": title,
        "text": text[:MAX_EXTRACT_CHARS],
    }
