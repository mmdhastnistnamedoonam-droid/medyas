"""Reads public Eitaa channel pages (https://eitaa.com/<id>) - the same page a browser shows.

Selectors come from a working open-source scraper (.js-widget_message_wrap / data-post /
.js-message_text / photo wrap with background-image). Only public channels with an ID work.
"""
import re
from html.parser import HTMLParser

import httpx

from . import config

BASE = "https://eitaa.com"
_VOID = {"br", "img", "input", "meta", "link", "hr", "source", "area", "base", "col", "embed", "wbr"}
_GENERIC = {"", "eitaa", "ایتا", "پیام رسان ایتا", "پیام‌رسان ایتا", "eitaa messenger"}
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:140.0) Gecko/20100101 Firefox/140.0"


class Node:
    __slots__ = ("tag", "attrs", "children", "parent")

    def __init__(self, tag: str, attrs: dict, parent=None):
        self.tag, self.attrs, self.children, self.parent = tag, attrs, [], parent

    def classes(self) -> set[str]:
        return set((self.attrs.get("class") or "").split())

    def walk(self):
        for c in self.children:
            if isinstance(c, Node):
                yield c
                yield from c.walk()

    def find_all(self, cls: str, tag: str | None = None):
        return [n for n in self.walk() if cls in n.classes() and (tag is None or n.tag == tag)]

    def find(self, cls: str, tag: str | None = None):
        r = self.find_all(cls, tag)
        return r[0] if r else None

    def text(self) -> str:
        out = []
        for c in self.children:
            if isinstance(c, str):
                out.append(c)
            elif c.tag == "br":
                out.append("\n")
            else:
                out.append(c.text())
        return "".join(out)


class _Builder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("root", {})
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        n = Node(tag, {k: (v or "") for k, v in attrs}, self.cur)
        self.cur.children.append(n)
        if tag not in _VOID:
            self.cur = n

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(Node(tag, {k: (v or "") for k, v in attrs}, self.cur))

    def handle_endtag(self, tag):
        n = self.cur
        while n is not self.root and n.tag != tag:
            n = n.parent
        if n is not self.root:
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.children.append(data)


def parse_dom(html: str) -> Node:
    b = _Builder()
    b.feed(html)
    b.close()
    return b.root


def _bg_url(style: str) -> str:
    m = re.search(r"url\((['\"]?)(.*?)\1\)", style)
    return m.group(2) if m else ""


def _clean(t: str) -> str:
    t = t.replace("\xa0", " ")
    t = re.sub(r"[ \t]+\n", "\n", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def parse_channel(html: str, username: str) -> dict:
    """-> {title, posts: [{key, text, images, time, forwarded}]}"""
    root = parse_dom(html)
    title = ""
    for cls in ("etme_channel_info_header_title", "etme_channel_info_title"):
        n = root.find(cls)
        if n:
            title = _clean(n.text())
            break
    if not title:
        for n in root.walk():
            if n.tag == "meta" and n.attrs.get("property") == "og:title":
                title = n.attrs.get("content", "").strip()
                break
    if not title:
        for n in root.walk():
            if n.tag == "title":
                title = _clean(n.text())
                break
    title = re.split(r"\s[-–|]\s*(?:Eitaa|ایتا)\s*$", title, flags=re.I)[0].strip()
    if title.lower() in _GENERIC:
        title = ""

    posts = []
    for wrap in root.find_all("js-widget_message_wrap"):
        m = wrap.find("js-widget_message")
        if not m or not m.attrs.get("data-post"):
            continue
        t = m.find("js-message_text")
        images = []
        for a in m.find_all("etme_widget_message_photo_wrap", "a"):
            u = _bg_url(a.attrs.get("style", ""))
            if u and "/download_" in u:
                full = u if u.startswith("http") else BASE + u
                if full not in images:
                    images.append(full)
        tm = next((n for n in m.walk() if n.tag == "time"), None)
        posts.append({
            "key": m.attrs["data-post"],
            "text": _clean(t.text()) if t else "",
            "images": images,
            "time": (tm.attrs.get("datetime") or "") if tm else "",
            "forwarded": bool(m.find("etme_widget_message_forwarded_from")),
        })
    return {"title": title, "posts": posts}


_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{2,63}$")


def normalize_username(raw: str) -> str:
    """'@name', 'eitaa.com/name', 'https://eitaa.com/name/123' -> 'name' ('' if not a channel id)."""
    s = raw.strip().strip("<>").strip()
    s = re.sub(r"^https?://", "", s, flags=re.I)
    s = re.sub(r"^(?:www\.)?eitaa\.com/", "", s, flags=re.I)
    s = s.lstrip("@").split("?")[0].split("#")[0].strip("/").split("/")[0]
    if s.lower() in {"joinchat", "s"} or not _ID_RE.match(s):
        return ""
    return s


async def fetch_channel(username: str) -> dict | None:
    """None = could not reach Eitaa. Otherwise {title, posts}; empty title = channel not found."""
    kw = {"proxy": config.EITAA_PROXY} if config.EITAA_PROXY else {}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20, connect=10), follow_redirects=True,
                                     headers={"User-Agent": _UA, "Accept-Language": "fa,en;q=0.7"}, **kw) as c:
            r = await c.get(f"{BASE}/{username}")
    except httpx.HTTPError:
        return None
    if r.status_code == 404:
        return {"title": "", "posts": []}
    if r.status_code != 200:
        return None
    return parse_channel(r.text, username)


async def download_image(url: str) -> bytes | None:
    kw = {"proxy": config.EITAA_PROXY} if config.EITAA_PROXY else {}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10), follow_redirects=True,
                                     headers={"User-Agent": _UA, "Referer": BASE + "/"}, **kw) as c:
            r = await c.get(url)
    except httpx.HTTPError:
        return None
    return r.content if r.status_code in (200, 206) and r.content else None
