"""附件處理：把 GCS 上的檔案轉成 LLM 可讀的 message content。"""
import asyncio
import base64
import logging
import os
from typing import Optional

import store

logger = logging.getLogger("router.attachments")

TEXT_MIMES = {"text/", "application/json", "application/xml", "application/javascript",
              "application/x-python", "application/x-sh"}

# Gemini 支援直接送 base64 的格式
INLINE_MIMES = {
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "application/pdf",
    "audio/wav", "audio/mp3", "audio/mpeg", "audio/aiff", "audio/aac", "audio/ogg", "audio/flac",
    "video/mp4", "video/mpeg", "video/mov", "video/avi", "video/webm", "video/3gpp",
}

OFFICE_MIMES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",   # .docx
    "application/vnd.openxmlformats-officedocument.presentationml.presentation", # .pptx
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",         # .xlsx
}

TEXT_EXTRACT_LIMIT = 50000


def is_text_mime(mime: str) -> bool:
    return any(mime.startswith(t) if t.endswith("/") else mime == t for t in TEXT_MIMES)


def extract_office_text(file_bytes: bytes, mime: str) -> str:
    import xml.etree.ElementTree as ET
    import zipfile
    from io import BytesIO
    try:
        with zipfile.ZipFile(BytesIO(file_bytes)) as z:
            names = z.namelist()
            parts = []

            if "word/document.xml" in names:  # docx
                with z.open("word/document.xml") as f:
                    tree = ET.parse(f)
                    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    parts = [t.text or "" for t in tree.findall(".//w:t", ns)]

            elif any(n.startswith("ppt/slides/slide") for n in names):  # pptx
                slides = sorted(n for n in names if n.startswith("ppt/slides/slide") and n.endswith(".xml"))
                for slide in slides:
                    with z.open(slide) as f:
                        tree = ET.parse(f)
                        ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
                        parts += [t.text or "" for t in tree.findall(".//a:t", ns)]

            elif "xl/sharedStrings.xml" in names:  # xlsx
                with z.open("xl/sharedStrings.xml") as f:
                    tree = ET.parse(f)
                    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                    parts = [t.text or "" for t in tree.findall(".//s:t", ns)]

            return "\n".join(p for p in parts if p.strip())
    except Exception:
        return ""


async def build_user_content(message: str, file_gcs_path: Optional[str],
                             file_mime_type: Optional[str]):
    """把使用者訊息與附件組成 LLM message content（str 或 multimodal list）。"""
    if not file_gcs_path or not store.gcs_ready:
        return message
    try:
        file_bytes, _ = await store.gcs_download(file_gcs_path)
        mime = (file_mime_type or "application/octet-stream").lower()

        # 純文字類型：直接附在訊息裡
        if is_text_mime(mime):
            text_content = file_bytes.decode("utf-8", errors="replace")
            return f"{message}\n\n```\n{text_content[:TEXT_EXTRACT_LIMIT]}\n```"

        # Office Open XML（.docx / .pptx / .xlsx）：解析文字
        if mime in OFFICE_MIMES:
            doc_text = await asyncio.to_thread(extract_office_text, file_bytes, mime)
            if doc_text:
                return f"{message}\n\n以下是文件內容：\n\n{doc_text[:TEXT_EXTRACT_LIMIT]}"
            return message

        # 模型原生支援的二進位格式（圖片、PDF、音訊、影片）
        if mime in INLINE_MIMES:
            b64 = base64.b64encode(file_bytes).decode()
            return [
                {"type": "text", "text": message},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ]

        # 不支援的格式（.doc/.ppt/.xls 舊格式等）
        ext = os.path.splitext(file_gcs_path)[1].upper() or mime
        return f"{message}\n\n（系統無法解析 {ext} 格式，請改用 .docx、.pptx、.xlsx、PDF 或圖片）"
    except Exception:
        logger.exception("附件處理失敗：%s", file_gcs_path)
        return message
