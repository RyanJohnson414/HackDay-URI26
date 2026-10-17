"""Fetch and extract readable text from a public web page."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from app.services.document_parser import ParsedDocument, ParsedPage

MAX_RESPONSE_BYTES = 1_500_000
MAX_REDIRECTS = 5
REQUEST_TIMEOUT_SECONDS = 12


class WebPageReadError(ValueError):
    """Invalid URL, failed fetch, or page with no extractable text."""


class _TextExtractor(HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "template"}
    _BLOCK_TAGS = {"address", "article", "blockquote", "br", "div", "footer", "h1", "h2",
                   "h3", "h4", "header", "li", "main", "p", "section", "td", "th", "tr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if not self._skip_depth and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if not self._skip_depth and tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)
        if not self._skip_depth:
            self.parts.append(data)

    def result(self) -> tuple[str, str]:
        lines = [" ".join(line.split()) for line in "".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if line), " ".join(" ".join(self.title_parts).split())


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class WebPage:
    url: str
    title: str
    text: str
    document: ParsedDocument


def _validate_public_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            raise WebPageReadError("URL must use http or https and include a hostname")
        if parts.username or parts.password:
            raise WebPageReadError("URLs with embedded credentials are not allowed")
        host = parts.hostname.rstrip(".")
        try:
            addresses = {ipaddress.ip_address(host)}
        except ValueError:
            try:
                addresses = {ipaddress.ip_address(item[4][0])
                             for item in socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80),
                                                            type=socket.SOCK_STREAM)}
            except OSError as exc:
                raise WebPageReadError(f"Could not resolve URL host: {exc}") from exc
        if not addresses or any(not address.is_global for address in addresses):
            raise WebPageReadError("Only publicly routable web hosts are allowed")
        return url
    except ValueError as exc:
        if isinstance(exc, WebPageReadError):
            raise
        raise WebPageReadError(f"Invalid URL: {exc}") from exc


def fetch_webpage(url: str, *, timeout: float = REQUEST_TIMEOUT_SECONDS,
                  max_bytes: int = MAX_RESPONSE_BYTES) -> WebPage:
    """Fetch public HTML and return extracted text as a citation-ready ParsedDocument."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    current_url = _validate_public_url(url)
    opener = build_opener(_NoRedirect)
    for redirect_count in range(MAX_REDIRECTS + 1):
        request = Request(current_url, headers={"User-Agent": "HaURI26/1.0 (document research prototype)",
                                                 "Accept": "text/html, text/plain;q=0.8"})
        try:
            response = opener.open(request, timeout=timeout)
        except HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308} and exc.headers.get("Location"):
                if redirect_count == MAX_REDIRECTS:
                    raise WebPageReadError("Page exceeded redirect limit") from exc
                current_url = _validate_public_url(urljoin(current_url, exc.headers["Location"]))
                continue
            raise WebPageReadError(f"Web server returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise WebPageReadError(f"Could not fetch URL: {exc}") from exc

        with response:
            final_url = _validate_public_url(response.geturl())
            content_type = response.headers.get_content_type().lower()
            if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
                raise WebPageReadError(f"Unsupported page content type: {content_type}")
            declared_length = response.headers.get("Content-Length")
            if declared_length and int(declared_length) > max_bytes:
                raise WebPageReadError(f"Page exceeds {max_bytes} byte limit")
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise WebPageReadError(f"Page exceeds {max_bytes} byte limit")
            charset = response.headers.get_content_charset() or "utf-8"
        try:
            markup = body.decode(charset, errors="replace")
        except LookupError as exc:
            raise WebPageReadError(f"Unsupported page character encoding: {charset}") from exc
        if content_type == "text/plain":
            text, title = markup.strip(), ""
        else:
            parser = _TextExtractor()
            parser.feed(markup)
            text, title = parser.result()
        if not text:
            raise WebPageReadError("Page contains no readable text")
        document = ParsedDocument(filename=title or final_url, media_type=content_type,
                                  pages=(ParsedPage(1, text),), source_url=final_url)
        return WebPage(final_url, title, text, document)
    raise WebPageReadError("Page could not be fetched")
