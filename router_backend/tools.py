"""搜尋工具：Tavily 網路／校內搜尋、北大官網抓取。"""
import datetime
import html as _htmllib
import re
import urllib.parse

import httpx

from config import TAVILY_KEY
from schemas import ChatRequest

# ------------------------------------------------------------------
# 工具 schema（給 LLM 的 function calling 定義）
# ------------------------------------------------------------------
NTPU_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_ntpu_web",
        "description": (
            "搜尋國立臺北大學（NTPU）官方網站（ntpu.edu.tw）內的資訊。"
            "當使用者詢問北大的公告、最新消息、課程、系所、行政規定、活動、"
            "師資或校內任何事務時使用。與北大無關的一般問題不要呼叫此工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜尋關鍵字，可用中文"},
                "latest": {
                    "type": "boolean",
                    "description": "使用者想要『最新／近期』的公告或消息（重視時間新舊）時設為 true，"
                                   "會改依發布日期由新到舊排序；一般查詢維持 false。",
                },
            },
            "required": ["query"],
        },
    },
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "搜尋即時網路資訊。當使用者詢問需要最新或即時的一般資訊"
            "（新聞、天氣、活動、非北大的時事等）時使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜尋關鍵字"},
            },
            "required": ["query"],
        },
    },
}

FETCH_PAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_ntpu_page",
        "description": (
            "打開指定的臺北大學（ntpu.edu.tw）網頁並讀取整頁全文。當你已從搜尋結果拿到某個北大網址、"
            "但摘要不足以回答使用者問題時，用此工具讀取該頁完整內容。只接受 ntpu.edu.tw 網域的網址。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "要讀取的臺北大學頁面完整網址（需為 ntpu.edu.tw 網域）"},
            },
            "required": ["url"],
        },
    },
}

NEWS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_ntpu_news",
        "description": (
            "取得臺北大學首頁的最新消息列表（全校各單位的公告、新聞、活動），依網站排序（通常最新在前）。"
            "當使用者想瀏覽最新／近期消息，或不確定該用什麼關鍵字搜尋時使用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "要回傳幾則，預設 15，最多 30"},
            },
        },
    },
}


def build_search_tools(req: ChatRequest) -> list:
    tools = []
    if req.ntpu_search_enabled:
        tools += [NTPU_SEARCH_TOOL, FETCH_PAGE_TOOL, NEWS_TOOL]
    if req.search_enabled:
        tools.append(WEB_SEARCH_TOOL)
    return tools


# ------------------------------------------------------------------
# Web search (Tavily)
# ------------------------------------------------------------------
async def web_search(query: str, count: int = 5, ntpu_only: bool = False,
                     sort_by_date: bool = False) -> dict:
    """Returns {"status": "ok"|"empty"|"quota"|"error", "text": str}."""
    if not TAVILY_KEY:
        return {"status": "error", "text": ""}
    try:
        payload: dict = {
            "api_key": TAVILY_KEY,
            "query": query,
            "max_results": count,
            "include_answer": True,
            "search_depth": "advanced",
        }
        if ntpu_only:
            payload["include_domains"] = ["ntpu.edu.tw"]
        else:
            payload["days"] = 30
        if sort_by_date:
            payload["time_range"] = "year"  # 偏向近一年，提升「最新公告」查詢的新鮮度
        async with httpx.AsyncClient() as c:
            resp = await c.post(
                "https://api.tavily.com/search",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=20,
            )
            if resp.status_code in (401, 403, 429):
                try:
                    msg = str(resp.json()).lower()
                    if any(k in msg for k in ("quota", "limit", "exceeded", "plan")):
                        return {"status": "quota", "text": ""}
                except Exception:
                    pass
                return {"status": "error", "text": ""}
            resp.raise_for_status()
            data = resp.json()
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            lines = [f"（搜尋日期：{today}）"]
            if data.get("answer"):
                lines.append(f"摘要：{data['answer']}")
            results = data.get("results", [])
            if sort_by_date:
                # Tavily 預設依相似度排序；抓最新公告時改依發布日期新→舊，無日期者排最後
                results = sorted(results, key=lambda r: r.get("published_date") or "", reverse=True)
            found = 0
            for r in results[:count]:
                title   = r.get("title", "")
                content = r.get("content", "")[:300]
                url     = r.get("url", "")
                pub     = r.get("published_date", "")
                date_str = f" [{pub}]" if pub else ""
                if title:
                    lines.append(f"• {title}{date_str}\n  {content}\n  {url}")
                    found += 1
            # 只有摘要（answer）但沒有任何實際結果時，視為查無資料，避免 AI 拿空殼編造
            if found == 0:
                return {"status": "empty", "text": ""}
            return {"status": "ok", "text": "\n\n".join(lines)}
    except Exception:
        return {"status": "error", "text": ""}


# ------------------------------------------------------------------
# 北大主站抓取（httpx；主站為 Angular SSR，新聞/文章頁內容在 HTML 內）
# ------------------------------------------------------------------
_NTPU_HOME = "https://new.ntpu.edu.tw/"
_BROWSER_UA = "Mozilla/5.0 (compatible; NTPU-AI/1.0)"


def _is_ntpu_url(url: str) -> bool:
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return host == "ntpu.edu.tw" or host.endswith(".ntpu.edu.tw")


def _html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", _htmllib.unescape(raw)).strip()


async def _fetch_ntpu(url: str) -> str:
    if not _is_ntpu_url(url):
        return "（只能讀取 ntpu.edu.tw 網域的頁面）"
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20,
                                     headers={"User-Agent": _BROWSER_UA}) as c:
            resp = await c.get(url)
            resp.raise_for_status()
            raw = resp.text
            final_url = str(resp.url)
    except Exception:
        return f"（無法開啟該頁面：{url}）"
    text = _html_to_text(raw)
    if len(text) < 40:
        return f"（該頁面沒有可讀取的文字內容，可能需要瀏覽器才能顯示：{final_url}）"
    m = re.search(r"發布日期[：: ]*([0-9]{4}\s*/\s*[0-9]{1,2}\s*/\s*[0-9]{1,2})", raw)
    head = f"來源：{final_url}\n"
    if m:
        clean_date = re.sub(r"\s+", "", m.group(1))
        head += f"發布日期：{clean_date}\n"
    return head + "\n" + text[:6000]


async def _ntpu_latest_news(count: int = 15) -> str:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20,
                                     headers={"User-Agent": _BROWSER_UA}) as c:
            resp = await c.get(_NTPU_HOME)
            resp.raise_for_status()
            raw = resp.text
    except Exception:
        return "（無法取得臺北大學最新消息）"
    anchors = re.findall(r'<a[^>]+href="([^"]*?/news/[^"]+)"[^>]*>(.*?)</a>', raw, re.S | re.I)
    seen: set = set()
    items: list = []
    for href, inner in anchors:
        parts = [p for p in urllib.parse.urlparse(href).path.split("/") if p]
        if len(parts) < 2:
            continue
        nid = parts[1]
        if nid in seen or nid == "news":
            continue
        title_from_url = urllib.parse.unquote(parts[-1]) if len(parts) >= 3 else ""
        inner_text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", inner)).strip()
        title = inner_text or title_from_url
        if not title or title.startswith("SYSTEM."):
            continue
        seen.add(nid)
        full = href if href.startswith("http") else "https://new.ntpu.edu.tw/" + href.lstrip("/")
        items.append(f"• {title}\n  {full}")
        if len(items) >= count:
            break
    if not items:
        return "（目前抓不到最新消息列表）"
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return (f"（臺北大學首頁最新消息，擷取日期 {today}，依網站排序，通常最新在前）\n\n"
            + "\n".join(items))


async def run_search_tool(name: str, args: dict) -> str:
    """依工具名稱執行對應搜尋，回傳給 LLM 的文字結果。"""
    if name == "fetch_ntpu_page":
        return await _fetch_ntpu((args.get("url") or "").strip())
    if name == "get_ntpu_news":
        try:
            n = int(args.get("count") or 15)
        except Exception:
            n = 15
        return await _ntpu_latest_news(max(1, min(30, n)))
    query = (args.get("query") or "").strip()
    if not query:
        return "（未提供搜尋關鍵字）"
    if name == "search_ntpu_web":
        sr = await web_search(query, count=8, ntpu_only=True, sort_by_date=bool(args.get("latest")))
        scope = "臺北大學官網"
    elif name == "search_web":
        sr = await web_search(query, count=5)
        scope = "網路"
    else:
        return "（未知的工具）"
    if sr["status"] == "ok":
        return sr["text"]
    if sr["status"] == "quota":
        return f"（{scope}搜尋額度已用盡，暫時無法查詢，請稍後再試）"
    return f"（在{scope}查無「{query}」的相關資料）"
