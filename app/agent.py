"""AI agent: (1) a chat assistant with tools, (2) an autopilot that reads the user's source
channels, drops ads / irrelevant posts and publishes (or proposes) the rest to the user's channel."""
import asyncio
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from html import escape

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy import delete, func, select

from . import ai, sources
from .db import (AgentCfg, AiUsage, Candidate, Channel, ChatMsg, Post, SeenPost, Session, Source,
                 User, get_settings, parse_ids, utcnow)

log = logging.getLogger("agent")

MAX_SOURCES = 30
MAX_AGE_H = 48
BATCH = 12
HISTORY = 12
_running: set[int] = set()


# ------------------------------------------------------------------ helpers
async def get_cfg(uid: int) -> AgentCfg:
    async with Session() as s:
        c = await s.get(AgentCfg, uid)
        if not c:
            c = AgentCfg(user_id=uid)
            s.add(c)
            await s.commit()
        return c


async def save_cfg(uid: int, **kw) -> AgentCfg:
    async with Session() as s:
        c = await s.get(AgentCfg, uid) or AgentCfg(user_id=uid)
        for k, v in kw.items():
            setattr(c, k, v)
        s.add(c)
        await s.commit()
        return c


async def user_sources(uid: int) -> list[Source]:
    async with Session() as s:
        return list((await s.execute(select(Source).where(Source.user_id == uid).order_by(Source.id))).scalars())


async def user_channels(uid: int) -> list[Channel]:
    async with Session() as s:
        return list((await s.execute(select(Channel).where(Channel.user_id == uid).order_by(Channel.id))).scalars())


async def use_quota(uid: int, is_admin: bool) -> bool:
    """Counts one AI request for today; False when the daily limit is used up."""
    if is_admin:
        return True
    lim = (await get_settings())["ai_daily_limit"].strip()
    limit = int(lim) if lim.isdigit() else 0
    day = utcnow().strftime("%Y-%m-%d")
    async with Session() as s:
        row = (await s.execute(select(AiUsage).where(AiUsage.user_id == uid, AiUsage.day == day))).scalar_one_or_none()
        if not row:
            row = AiUsage(user_id=uid, day=day, n=0)
            s.add(row)
        if limit and row.n >= limit:
            return False
        row.n += 1
        await s.commit()
    return True


async def add_sources(uid: int, raw: str) -> list[dict]:
    """One line per channel (id or link). Saves the ones that exist; returns a result per line."""
    lines = [ln.strip() for ln in raw.replace("،", "\n").splitlines() if ln.strip()][:15]
    have = {s.username.lower() for s in await user_sources(uid)}
    results = []
    for ln in lines:
        name = sources.normalize_username(ln)
        r = {"raw": ln, "username": name, "status": "invalid", "title": "", "n": 0}
        if not name:
            r["status"] = "private" if "joinchat" in ln else "invalid"
        elif name.lower() in have:
            r["status"] = "dup"
        elif len(have) >= MAX_SOURCES:
            r["status"] = "limit"
        else:
            data = await sources.fetch_channel(name)
            if data is None:
                r["status"] = "neterr"
            elif not data["title"] and not data["posts"]:
                r["status"] = "notfound"
            else:
                r.update(title=data["title"] or name, n=len(data["posts"]))
                if data["posts"]:
                    r["status"] = "ok"
                    async with Session() as s:
                        s.add(Source(user_id=uid, username=name, title=r["title"][:128]))
                        await s.commit()
                    have.add(name.lower())
                else:
                    r["status"] = "noposts"
            await asyncio.sleep(0.8)
        results.append(r)
    return results


async def remove_source(uid: int, sid: int) -> None:
    async with Session() as s:
        await s.execute(delete(Source).where(Source.id == sid, Source.user_id == uid))
        await s.commit()


async def queue_post(uid: int, channel_id: int, text: str, image_url: str = "") -> int:
    text = text.strip()
    with_img = bool(image_url) and len(text) <= 1000 and len(image_url) <= 240
    async with Session() as s:
        p = Post(user_id=uid, channel_id=channel_id, kind="file" if with_img else "text",
                 text=text[:4000], file_id=f"url:{image_url}"[:256] if with_img else "",
                 file_name="image.jpg" if with_img else "", run_at=utcnow())
        s.add(p)
        await s.commit()
        return p.id


def _age_ok(t: str) -> bool:
    if not t:
        return True
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        return True
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return utcnow() - dt <= timedelta(hours=MAX_AGE_H)


def _key(src: str, post_key: str) -> str:
    return f"{src.lower()}|{post_key}"[:160]


# ------------------------------------------------------------------ autopilot
_JUDGE_SYS = """تو سردبیر یک کانال هستی. چند پست از کانال‌های منبع به تو داده می‌شود. برای هر کدام تصمیم بگیر که در کانال کاربر منتشر شود یا نه.
منتشر نکن اگر: تبلیغ، معرفی محصول/خدمات/فروش، دعوت به عضویت یا معرفی کانال دیگر، کلیک‌بیت، اسپانسری، متن بی‌محتوا، یا با موضوع و قوانین کانال کاربر نمی‌خواند، یا شبیه پست دیگری در همین فهرست است.
منتشر کن فقط اگر محتوای واقعی و مفید برای مخاطبان کانال کاربر دارد.
اگر دستور «بازنویسی» روشن است، متن را به فارسی روان و با کلمات خودت بازنویسی کن، معنی را حفظ کن، و هر لینک، آیدی (@...)، نام کانال منبع و امضا را حذف کن. اگر خاموش است، همان متن اصلی را (فقط بدون لینک و آیدی منبع) بده.
فقط و فقط یک JSON معتبر برگردان، بدون هیچ متن دیگر، به این شکل:
{"items":[{"i":0,"publish":true,"reason":"دلیل کوتاه","text":"متن نهایی"}]}
برای هر ایندکس یک آیتم بده. اگر publish=false بود text لازم نیست."""


def _extract_json(s: str) -> dict:
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no json")
    return json.loads(s[a:b + 1])


def _approve_kb(cid: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="انتشار", callback_data=f"cd:ok:{cid}"),
        InlineKeyboardButton(text="رد", callback_data=f"cd:no:{cid}")]])


async def curate(uid: int, manager, *, is_admin: bool = False) -> str:
    cfg = await get_settings()
    if not ai.configured(cfg):
        return "هوش مصنوعی هنوز توسط مدیر فعال نشده است."
    ag = await get_cfg(uid)
    async with Session() as s:
        ch = await s.get(Channel, ag.channel_id) if ag.channel_id else None
    if not ch or ch.user_id != uid:
        return "کانال مقصد انتخاب نشده است."
    srcs = await user_sources(uid)
    if not srcs:
        return "هنوز کانال محتوایی اضافه نکرده‌اید."
    if not await use_quota(uid, is_admin):
        return "سهمیه‌ی امروز هوش مصنوعی تمام شده است."

    async with Session() as s:
        seen = set((await s.execute(select(SeenPost.post_key).where(SeenPost.user_id == uid))).scalars())
    fresh, to_mark = [], []
    for src in srcs:
        data = await sources.fetch_channel(src.username)
        await asyncio.sleep(1)
        if not data:
            continue
        for p in data["posts"]:
            k = _key(src.username, p["key"])
            if k in seen:
                continue
            if (not p["text"] and not p["images"]) or not _age_ok(p["time"]):
                to_mark.append(k)
                continue
            fresh.append({**p, "src": src.username, "k": k})
    fresh.sort(key=lambda p: p["time"], reverse=True)
    batch, rest = fresh[:BATCH], fresh[BATCH:]
    chosen = []
    summary = f"{len(batch)} پست جدید بررسی شد"
    if batch:
        listing = "\n\n".join(f"[{i}] منبع: {p['src']}{' | دارای عکس' if p['images'] else ''}\n{p['text'][:700] or '(بدون متن)'}"
                              for i, p in enumerate(batch))
        user_msg = (f"موضوع و قوانین کانال کاربر:\n{ag.brief or '(مشخص نشده؛ محتوای عمومی و مفید)'}\n\n"
                    f"بازنویسی: {'روشن' if ag.rewrite else 'خاموش'}\n\nپست‌ها:\n{listing}")
        try:
            resp = await ai.chat(cfg, _JUDGE_SYS, [{"role": "user", "content": user_msg}], max_tokens=3000)
            items = {int(it["i"]): it for it in _extract_json(resp["text"]).get("items", []) if "i" in it}
        except (ai.AIError, ValueError, KeyError, TypeError) as e:
            log.warning("curate uid=%s failed: %s", uid, e)
            await save_cfg(uid, last_run=utcnow())
            return f"بررسی با هوش مصنوعی ناموفق بود: {str(e)[:120]}"
        to_mark += [p["k"] for p in batch]
        for i, p in enumerate(batch):
            it = items.get(i)
            if it and it.get("publish") and len(chosen) < max(1, ag.max_per_run):
                text = str(it.get("text") or p["text"]).strip()
                if text:
                    chosen.append((p, text))
        summary += f"، {len(chosen)} مورد انتخاب شد"
    async with Session() as s:
        for k in set(to_mark):
            s.add(SeenPost(user_id=uid, post_key=k))
        await s.commit()
    await save_cfg(uid, last_run=utcnow())

    for p, text in chosen:
        img = p["images"][0] if p["images"] else ""
        if ag.auto:
            pid = await queue_post(uid, ch.id, text, img)
            note = f"عامل یک پست از «{escape(p['src'])}» را در صف انتشار گذاشت (#{pid}):\n\n{escape(text[:600])}"
            kb = None
        else:
            async with Session() as s:
                c = Candidate(user_id=uid, channel_id=ch.id, source=p["src"], text=text[:4000], image_url=img)
                s.add(c)
                await s.commit()
            note = f"پیشنهاد عامل (از «{escape(p['src'])}»):\n\n{escape(text[:900])}"
            kb = _approve_kb(c.id)
        if manager.bot:
            try:
                await manager.bot.send_message(uid, note, reply_markup=kb)
            except Exception:
                log.warning("notify failed uid=%s", uid)
    return summary + (f" ({len(rest)} پست دیگر نوبت بعد)" if rest else "")


async def agent_loop(manager) -> None:
    log.info("agent loop started")
    while True:
        try:
            await asyncio.sleep(60)
            now = utcnow()
            admins = parse_ids((await get_settings())["admin_ids"])
            async with Session() as s:
                rows = list((await s.execute(select(AgentCfg).where(
                    AgentCfg.enabled.is_(True), AgentCfg.channel_id > 0))).scalars())
                users = {u.id: u for u in (await s.execute(select(User).where(
                    User.id.in_([r.user_id for r in rows] or [0])))).scalars()}
            for r in rows:
                u = users.get(r.user_id)
                ok = r.user_id in admins or (u and not u.banned and u.expires_at and u.expires_at > now)
                due = r.last_run is None or now >= r.last_run + timedelta(hours=max(1, r.interval_h))
                if ok and due and r.user_id not in _running:
                    _running.add(r.user_id)
                    try:
                        await curate(r.user_id, manager, is_admin=r.user_id in admins)
                    except Exception:
                        log.exception("curate failed uid=%s", r.user_id)
                    finally:
                        _running.discard(r.user_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("agent loop error")


# ------------------------------------------------------------------ chat agent
TOOLS = [
    {"name": "list_sources", "description": "فهرست کانال‌های محتوای (منبع) کاربر در ایتا.",
     "schema": {"type": "object", "properties": {}}},
    {"name": "read_source", "description": "آخرین پست‌های یک کانال عمومی ایتا را می‌خواند (متن و تعداد عکس).",
     "schema": {"type": "object", "properties": {
         "username": {"type": "string", "description": "آیدی کانال ایتا بدون @"},
         "limit": {"type": "integer", "description": "حداکثر تعداد پست (پیش‌فرض ۸)"}}, "required": ["username"]}},
    {"name": "list_my_channels", "description": "کانال‌های خود کاربر که می‌توان در آن‌ها پست گذاشت (ایتا/روبیکا).",
     "schema": {"type": "object", "properties": {}}},
    {"name": "publish_post",
     "description": "یک پست متنی (و اختیاری یک عکس از آدرس) را در صف انتشار یکی از کانال‌های کاربر می‌گذارد. فقط وقتی کاربر صریحاً خواسته.",
     "schema": {"type": "object", "properties": {
         "channel_id": {"type": "integer"}, "text": {"type": "string"},
         "image_url": {"type": "string", "description": "اختیاری؛ آدرس عکس یک پست منبع"}},
         "required": ["channel_id", "text"]}},
    {"name": "get_agent_settings", "description": "تنظیمات فعلی عامل خودکار را برمی‌گرداند.",
     "schema": {"type": "object", "properties": {}}},
    {"name": "update_agent_settings",
     "description": "تنظیمات عامل خودکار را عوض می‌کند. فقط فیلدهایی را بده که باید تغییر کنند.",
     "schema": {"type": "object", "properties": {
         "brief": {"type": "string", "description": "موضوع کانال و قوانین انتخاب محتوا"},
         "channel_id": {"type": "integer", "description": "کانال مقصد"},
         "enabled": {"type": "boolean"}, "auto": {"type": "boolean", "description": "انتشار بدون تأیید"},
         "rewrite": {"type": "boolean"}, "interval_hours": {"type": "integer"},
         "max_per_run": {"type": "integer"}}}},
    {"name": "run_curation_now", "description": "همین الان کانال‌های منبع را بررسی و پست‌های مناسب را انتخاب می‌کند.",
     "schema": {"type": "object", "properties": {}}},
]


def _cfg_dict(ag: AgentCfg) -> dict:
    return {"channel_id": ag.channel_id, "enabled": ag.enabled, "auto_publish": ag.auto, "rewrite": ag.rewrite,
            "interval_hours": ag.interval_h, "max_per_run": ag.max_per_run, "brief": ag.brief}


async def run_tool(uid: int, is_admin: bool, manager, name: str, a: dict) -> str:
    try:
        if name == "list_sources":
            return json.dumps([{"username": s.username, "title": s.title} for s in await user_sources(uid)],
                              ensure_ascii=False)
        if name == "read_source":
            u = sources.normalize_username(str(a.get("username", "")))
            data = await sources.fetch_channel(u) if u else None
            if not data or not data["posts"]:
                return "کانال پیدا نشد یا پستی قابل خواندن نداشت."
            n = max(1, min(int(a.get("limit") or 8), 15))
            return json.dumps([{"time": p["time"], "text": p["text"][:600], "images": len(p["images"]),
                                "image_url": p["images"][0] if p["images"] else ""}
                               for p in sorted(data["posts"], key=lambda x: x["time"], reverse=True)[:n]],
                              ensure_ascii=False)
        if name == "list_my_channels":
            return json.dumps([{"channel_id": c.id, "title": c.title, "platform": c.platform}
                               for c in await user_channels(uid)], ensure_ascii=False)
        if name == "publish_post":
            async with Session() as s:
                ch = await s.get(Channel, int(a["channel_id"]))
                u = await s.get(User, uid)
            if not ch or ch.user_id != uid:
                return "این کانال متعلق به کاربر نیست."
            if not (is_admin or (u and u.expires_at and u.expires_at > utcnow())):
                return "اشتراک کاربر فعال نیست."
            pid = await queue_post(uid, ch.id, str(a["text"]), str(a.get("image_url") or ""))
            return f"پست #{pid} در صف انتشار قرار گرفت."
        if name == "get_agent_settings":
            return json.dumps(_cfg_dict(await get_cfg(uid)), ensure_ascii=False)
        if name == "update_agent_settings":
            kw = {}
            if "brief" in a:
                kw["brief"] = str(a["brief"])[:2000]
            if "channel_id" in a:
                chans = {c.id for c in await user_channels(uid)}
                if int(a["channel_id"]) not in chans:
                    return "کانال مقصد معتبر نیست."
                kw["channel_id"] = int(a["channel_id"])
            for k_in, k_db in (("enabled", "enabled"), ("auto", "auto"), ("rewrite", "rewrite")):
                if k_in in a:
                    kw[k_db] = bool(a[k_in])
            if "interval_hours" in a:
                kw["interval_h"] = max(1, min(int(a["interval_hours"]), 72))
            if "max_per_run" in a:
                kw["max_per_run"] = max(1, min(int(a["max_per_run"]), 5))
            return "ذخیره شد: " + json.dumps(_cfg_dict(await save_cfg(uid, **kw)), ensure_ascii=False)
        if name == "run_curation_now":
            return await curate(uid, manager, is_admin=is_admin)
    except Exception as e:  # tool errors go back to the model, not to the user
        return f"خطا: {str(e)[:200]}"
    return "ابزار ناشناخته."


def _system(ag: AgentCfg, n_src: int, n_chan: int) -> str:
    return (
        "تو «دستیار هوشمند» یک ربات مدیریت کانال‌های ایتا و روبیکا هستی و با کاربر فارسی و کوتاه و دوستانه صحبت می‌کنی.\n"
        "کارهایت: گپ و ایده‌دادن برای محتوا؛ خواندن کانال‌های منبع در ایتا؛ جداکردن تبلیغ از محتوای مفید؛ "
        "نوشتن/بازنویسی پست؛ انتشار پست در کانال کاربر؛ و تنظیم عامل خودکار با ابزارها.\n"
        "قواعد: فقط وقتی کاربر صریحاً خواست پست منتشر کن. چیزی از خودت به‌عنوان پستِ منبع نساز و آنچه ابزار نگفته را حدس نزن. "
        "قبل از انتشار اگر کانال مقصد مشخص نیست از list_my_channels کمک بگیر. نتیجه‌ی ابزارها را صادقانه گزارش کن. "
        "از قالب‌بندی سنگین و جدول استفاده نکن.\n"
        f"وضعیت: {n_src} کانال منبع، {n_chan} کانال کاربر. عامل خودکار: {'روشن' if ag.enabled else 'خاموش'}.\n"
        f"موضوع/قوانین کانال کاربر: {ag.brief or 'هنوز مشخص نشده'}"
    )


async def chat_turn(uid: int, is_admin: bool, manager, text: str) -> str:
    cfg = await get_settings()
    if not ai.configured(cfg):
        return "دستیار هوشمند هنوز توسط مدیر فعال نشده است."
    if not await use_quota(uid, is_admin):
        return "سهمیه‌ی امروز دستیار تمام شده است؛ فردا دوباره امتحان کنید."
    ag = await get_cfg(uid)
    async with Session() as s:
        hist = list((await s.execute(select(ChatMsg).where(ChatMsg.user_id == uid)
                                     .order_by(ChatMsg.id.desc()).limit(HISTORY))).scalars())[::-1]
    msgs = [{"role": h.role, "content": h.content} for h in hist]
    msgs.append({"role": "user", "content": text})
    system = _system(ag, len(await user_sources(uid)), len(await user_channels(uid)))
    reply = ""
    try:
        for _ in range(6):
            r = await ai.chat(cfg, system, msgs, TOOLS, max_tokens=1500)
            if not r["tool_calls"]:
                reply = r["text"] or "…"
                break
            msgs.append({"role": "assistant", "content": r["text"], "tool_calls": r["tool_calls"]})
            for tc in r["tool_calls"]:
                out = await run_tool(uid, is_admin, manager, tc["name"], tc["input"])
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": out[:6000]})
        else:
            reply = "کار طولانی شد؛ لطفاً دوباره و ساده‌تر بگویید."
    except ai.AIError as e:
        return f"خطا از سرویس هوش مصنوعی: {e}"
    async with Session() as s:
        s.add(ChatMsg(user_id=uid, role="user", content=text[:4000]))
        s.add(ChatMsg(user_id=uid, role="assistant", content=reply[:4000]))
        await s.commit()
        old = (await s.execute(select(ChatMsg.id).where(ChatMsg.user_id == uid)
                               .order_by(ChatMsg.id.desc()).offset(40))).scalars().all()
        if old:
            await s.execute(delete(ChatMsg).where(ChatMsg.id.in_(old)))
            await s.commit()
    return reply


async def clear_chat(uid: int) -> None:
    async with Session() as s:
        await s.execute(delete(ChatMsg).where(ChatMsg.user_id == uid))
        await s.commit()
