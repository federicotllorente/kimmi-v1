"""Descarga y resume documentación oficial de librerías/APIs.

Solo hosts allowlisted. Extracto limpio + resumen corto vía Ollama.
Requiere: stdlib + ollama (para el resumen).
"""
from __future__ import annotations

import html
import ipaddress
import re
import socket
from html.parser import HTMLParser
from typing import List, Optional, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

MAX_RESPONSE_BYTES = 500_000
MAX_EXTRACT_CHARS = 4000
MAX_OUTPUT_CHARS = 2000
FETCH_TIMEOUT_SEC = 15
USER_AGENT = "JarvisDocsBot/1.0 (+local-assistant)"

ALLOWED_HOSTS: Set[str] = {
    "docs.python.org",
    "pypi.org",
    "readthedocs.io",
    "developer.mozilla.org",
    "nextjs.org",
    "nodejs.org",
    "docs.docker.com",
    "docs.npmjs.com",
    "react.dev",
    "fastapi.tiangolo.com",
    "docs.pytorch.org",
    "numpy.org",
    "pandas.pydata.org",
    "learn.microsoft.com",
    "code.visualstudio.com",
    "cursor.com",
    "www.cursor.com",
    "ollama.com",
    "github.com",
    "raw.githubusercontent.com",
}

# library short-name -> docs base URL
LIBRARY_DOCS = {
    "requests": "https://requests.readthedocs.io/en/latest/",
    "python": "https://docs.python.org/3/",
    "nextjs": "https://nextjs.org/docs",
    "next": "https://nextjs.org/docs",
    "react": "https://react.dev/reference/react",
    "docker": "https://docs.docker.com/",
    "nodejs": "https://nodejs.org/docs/latest/api/",
    "node": "https://nodejs.org/docs/latest/api/",
    "fastapi": "https://fastapi.tiangolo.com/",
    "numpy": "https://numpy.org/doc/stable/",
    "pandas": "https://pandas.pydata.org/docs/",
    "mdn": "https://developer.mozilla.org/en-US/docs/Web",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._skip_depth = 0
        self._skip_tags = {"script", "style", "noscript", "svg", "nav", "footer", "header"}

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        return "\n".join(self._parts)


def fetch_web_docs(
    url: str = "",
    library: str = "",
    topic: str = "",
    llm_model: str = "llama3.2:3b",
) -> str:
    """Obtiene docs desde URL o library+topic; resume con Ollama."""
    target = (url or "").strip()
    lib = (library or "").strip().lower()
    topic_s = (topic or "").strip()

    if not target:
        if not lib:
            return "Indica una url o un library (y opcionalmente topic)."
        base = LIBRARY_DOCS.get(lib)
        if not base:
            known = ", ".join(sorted(LIBRARY_DOCS.keys()))
            return f"Librería desconocida. Conocidas: {known}. O pasa una url allowlisted."
        if topic_s:
            # Búsqueda simple: unir topic al path / buscar en query
            safe_topic = re.sub(r"[^A-Za-z0-9._/-]+", "-", topic_s).strip("-")
            target = urljoin(base, safe_topic)
        else:
            target = base

    ok, reason = _validate_url(target)
    if not ok:
        return reason

    try:
        body, final_url = _fetch(target)
    except Exception as e:
        return f"No se pudo descargar la documentación: {e}"

    ok_final, reason_final = _validate_url(final_url)
    if not ok_final:
        return f"Redirect bloqueado: {reason_final}"

    extract = _html_to_text(body)
    if not extract.strip():
        return "La página no contiene texto extraíble."

    extract = extract[:MAX_EXTRACT_CHARS]
    summary = _summarize(extract, llm_model, topic_s or lib or final_url)
    output = f"Fuente: {final_url}\n\n{summary}"
    return _truncate(output)


def _validate_url(url: str) -> Tuple[bool, str]:
    try:
        parsed = urlparse(url)
    except Exception:
        return False, "URL inválida."

    if parsed.scheme not in ("http", "https"):
        return False, "Solo se permiten URLs http/https."

    host = (parsed.hostname or "").lower()
    if not host:
        return False, "URL sin host."

    if not _host_allowed(host):
        return False, f"Host no permitido: {host}."

    # Bloquear IPs privadas / localhost
    try:
        infos = socket.getaddrinfo(host, None)
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
            ):
                return False, "Se bloqueó una dirección IP privada o local."
    except socket.gaierror:
        return False, f"No se pudo resolver el host '{host}'."

    return True, ""


def _host_allowed(host: str) -> bool:
    if host in ALLOWED_HOSTS:
        return True
    # Permitir subdominios de readthedocs.io y similares
    for allowed in ALLOWED_HOSTS:
        if host.endswith("." + allowed):
            return True
    return False


def _fetch(url: str) -> Tuple[str, str]:
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain,*/*"})
    with urlopen(req, timeout=FETCH_TIMEOUT_SEC) as resp:
        final = resp.geturl()
        raw = resp.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raw = raw[:MAX_RESPONSE_BYTES]
        charset = "utf-8"
        ctype = resp.headers.get_content_charset()
        if ctype:
            charset = ctype
        text = raw.decode(charset, errors="replace")
        return text, final


def _html_to_text(body: str) -> str:
    # Si parece HTML, parsear; si no, devolver plano.
    if "<html" in body.lower() or "<body" in body.lower() or "<p" in body.lower():
        parser = _TextExtractor()
        try:
            parser.feed(body)
            text = parser.get_text()
        except Exception:
            text = re.sub(r"<[^>]+>", " ", body)
        text = html.unescape(text)
    else:
        text = body
    # Compactar blancos
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _summarize(extract: str, llm_model: str, focus: str) -> str:
    try:
        import ollama
    except ImportError:
        return "Extracto (sin resumen; falta ollama):\n" + extract[:1200]

    prompt = (
        "Resume en español, en 5 a 8 líneas cortas, la documentación siguiente. "
        "Sé concreto y técnico. Sin markdown ni listas con viñetas.\n"
    )
    if focus:
        prompt += f"Enfócate en: {focus}\n"
    prompt += "\n---\n" + extract[:3000]

    try:
        resp = ollama.chat(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            keep_alive="24h",
        )
        content = resp.get("message", {}).get("content", "") if isinstance(resp, dict) else ""
        if not content:
            content = getattr(getattr(resp, "message", None), "content", "") or ""
        content = str(content).strip()
        if content:
            return content
    except Exception as e:
        return f"Extracto (resumen falló: {e}):\n" + extract[:1200]

    return "Extracto:\n" + extract[:1200]


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + "... [truncado]"
    return text
