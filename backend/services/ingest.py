"""Turn a URL into clean article text (title + body + domain)."""
from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

from backend.core import config

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


@dataclass
class Article:
    url: str
    domain: str
    title: str | None
    text: str


def domain_of(url: str) -> str:
    host = urlparse(url).netloc.lower().split("@")[-1].split(":")[0]
    return host[4:] if host.startswith("www.") else host


def _is_public(host: str) -> bool:
    """Block SSRF against private / loopback addresses."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    return True


def fetch_article(url: str) -> Article:
    import trafilatura

    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("invalid URL")
    host = domain_of(url)
    if not _is_public(host):
        raise ValueError("URL host is not publicly reachable")

    resp = requests.get(url, headers={"User-Agent": UA}, timeout=config.FETCH_TIMEOUT,
                        allow_redirects=True)
    resp.raise_for_status()
    html = resp.text
    text = trafilatura.extract(html, include_comments=False, include_tables=False,
                               favor_precision=True) or ""
    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else None)
    if not title:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
        title = re.sub(r"\s+", " ", m.group(1)).strip() if m else None
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 80 and title:
        text = title + ". " + text
    if not text:
        raise ValueError("could not extract readable text from that page")
    return Article(url=url, domain=domain_of(resp.url or url), title=title, text=text[:12000])
