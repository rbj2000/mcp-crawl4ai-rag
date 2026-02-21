"""ADF (Atlassian Document Format) and XHTML/Storage → Markdown converters"""

import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ADFConverter:
    """Converts Atlassian Document Format (Cloud) JSON to Markdown."""

    def convert(self, adf_doc: dict) -> str:
        if not adf_doc or not isinstance(adf_doc, dict):
            return ""
        content = adf_doc.get("content", [])
        parts = [self._convert_block(node) for node in content]
        md = "\n\n".join(p for p in parts if p)
        return md.strip()

    def _convert_block(self, node: dict) -> str:
        ntype = node.get("type", "")
        attrs = node.get("attrs", {})
        content = node.get("content", [])

        if ntype == "heading":
            level = attrs.get("level", 1)
            text = self._convert_inline_children(content)
            return f"{'#' * level} {text}"

        if ntype == "paragraph":
            return self._convert_inline_children(content)

        if ntype == "codeBlock":
            language = attrs.get("language", "")
            text = self._extract_text(content)
            return f"```{language}\n{text}\n```"

        if ntype == "blockquote":
            inner = self._convert_children_blocks(content)
            return "\n".join(f"> {line}" for line in inner.split("\n"))

        if ntype == "rule":
            return "---"

        if ntype == "bulletList":
            return self._convert_list(content, ordered=False)

        if ntype == "orderedList":
            return self._convert_list(content, ordered=True)

        if ntype == "table":
            return self._convert_table(content)

        if ntype == "panel":
            panel_type = attrs.get("panelType", "info").capitalize()
            inner = self._convert_children_blocks(content)
            return f"> **{panel_type}:** {inner}"

        if ntype == "expand":
            title = attrs.get("title", "Details")
            inner = self._convert_children_blocks(content)
            return f"**{title}:** {inner}"

        if ntype == "mediaSingle" or ntype == "mediaGroup":
            return self._convert_children_blocks(content)

        if ntype == "media":
            alt = attrs.get("alt", "")
            url = attrs.get("url", "")
            return f"![{alt}]({url})"

        # Fallback: try to render children
        if content:
            return self._convert_children_blocks(content)
        return ""

    def _convert_inline_children(self, nodes: list) -> str:
        return "".join(self._convert_inline(n) for n in nodes)

    def _convert_inline(self, node: dict) -> str:
        ntype = node.get("type", "")

        if ntype == "text":
            text = node.get("text", "")
            marks = node.get("marks", [])
            return self._apply_marks(text, marks)

        if ntype == "hardBreak":
            return "\n"

        if ntype == "inlineCard":
            url = node.get("attrs", {}).get("url", "")
            return f"[{url}]({url})"

        if ntype == "mention":
            name = node.get("attrs", {}).get("text", "")
            if not name:
                name = node.get("attrs", {}).get("id", "unknown")
            return f"@{name}"

        if ntype == "emoji":
            return node.get("attrs", {}).get("shortName", "")

        # Inline nodes with content (e.g. status)
        content = node.get("content", [])
        if content:
            return self._convert_inline_children(content)
        return ""

    def _apply_marks(self, text: str, marks: list) -> str:
        for mark in marks:
            mtype = mark.get("type", "")
            if mtype == "strong":
                text = f"**{text}**"
            elif mtype == "em":
                text = f"*{text}*"
            elif mtype == "code":
                text = f"`{text}`"
            elif mtype == "link":
                href = mark.get("attrs", {}).get("href", "")
                text = f"[{text}]({href})"
            elif mtype == "strike":
                text = f"~~{text}~~"
        return text

    def _convert_list(self, items: list, ordered: bool, depth: int = 0) -> str:
        lines = []
        indent = "  " * depth
        for i, item in enumerate(items):
            if item.get("type") != "listItem":
                continue
            content = item.get("content", [])
            # First block is the item text, rest may be nested lists
            text_parts = []
            nested = []
            for child in content:
                ctype = child.get("type", "")
                if ctype in ("bulletList", "orderedList"):
                    nested.append(child)
                else:
                    text_parts.append(child)

            item_text = self._convert_children_blocks(text_parts)
            prefix = f"{i + 1}." if ordered else "-"
            lines.append(f"{indent}{prefix} {item_text}")

            for nest in nested:
                is_ordered = nest.get("type") == "orderedList"
                lines.append(self._convert_list(
                    nest.get("content", []), ordered=is_ordered, depth=depth + 1
                ))
        return "\n".join(lines)

    def _convert_table(self, rows: list) -> str:
        if not rows:
            return ""

        md_rows: List[List[str]] = []
        is_header_row = [False] * len(rows)

        for ri, row in enumerate(rows):
            if row.get("type") != "tableRow":
                continue
            cells = row.get("content", [])
            md_cells = []
            has_header = False
            for cell in cells:
                ctype = cell.get("type", "")
                text = self._convert_children_blocks(cell.get("content", []))
                text = text.replace("\n", " ").strip()
                md_cells.append(text)
                if ctype == "tableHeader":
                    has_header = True
            md_rows.append(md_cells)
            is_header_row[ri] = has_header

        if not md_rows:
            return ""

        # Normalize column count
        max_cols = max(len(r) for r in md_rows)
        for row in md_rows:
            while len(row) < max_cols:
                row.append("")

        lines = []
        lines.append("| " + " | ".join(md_rows[0]) + " |")
        lines.append("| " + " | ".join("---" for _ in range(max_cols)) + " |")
        for row in md_rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def _convert_children_blocks(self, nodes: list) -> str:
        parts = [self._convert_block(n) for n in nodes]
        return "\n\n".join(p for p in parts if p)

    def _extract_text(self, nodes: list) -> str:
        """Extract plain text from inline nodes (for code blocks)."""
        parts = []
        for node in nodes:
            if node.get("type") == "text":
                parts.append(node.get("text", ""))
            elif node.get("content"):
                parts.append(self._extract_text(node["content"]))
        return "".join(parts)


class XHTMLConverter:
    """Converts Confluence XHTML/Storage format to Markdown.

    Uses ``xml.etree.ElementTree`` — no external dependencies.
    """

    # Confluence macro namespace
    AC_NS = "http://atlassian.com/content"
    RI_NS = "http://atlassian.com/resource/identifier"

    def convert(self, xhtml: str) -> str:
        if not xhtml or not xhtml.strip():
            return ""
        # Wrap in root for valid XML; register namespaces
        wrapped = (
            f'<root xmlns:ac="{self.AC_NS}" xmlns:ri="{self.RI_NS}">'
            f"{xhtml}</root>"
        )
        try:
            root = ET.fromstring(wrapped)
        except ET.ParseError as e:
            logger.warning("XHTML parse error, falling back to text strip: %s", e)
            return self._strip_tags(xhtml)

        md = self._walk(root).strip()
        # Collapse excessive blank lines
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md

    def _walk(self, elem: ET.Element) -> str:
        tag = self._local_tag(elem.tag)
        return self._convert_element(elem, tag)

    def _convert_element(self, elem: ET.Element, tag: str) -> str:
        # --- Headings ---
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            text = self._inner_text(elem).strip()
            return f"\n\n{'#' * level} {text}\n\n"

        # --- Paragraph ---
        if tag == "p":
            text = self._inner_text(elem).strip()
            return f"\n\n{text}\n\n" if text else ""

        # --- Line break ---
        if tag == "br":
            return "\n"

        # --- Bold / Italic ---
        if tag in ("strong", "b"):
            return f"**{self._inner_text(elem)}**"
        if tag in ("em", "i"):
            return f"*{self._inner_text(elem)}*"

        # --- Code ---
        if tag == "code":
            return f"`{self._inner_text(elem)}`"
        if tag == "pre":
            return f"\n\n```\n{self._inner_text(elem)}\n```\n\n"

        # --- Links ---
        if tag == "a":
            href = elem.get("href", "")
            text = self._inner_text(elem).strip()
            return f"[{text}]({href})"

        # --- Images ---
        if tag == "img":
            src = elem.get("src", "")
            alt = elem.get("alt", "")
            return f"![{alt}]({src})"

        # --- Lists ---
        if tag == "ul":
            return "\n\n" + self._convert_list(elem, ordered=False) + "\n\n"
        if tag == "ol":
            return "\n\n" + self._convert_list(elem, ordered=True) + "\n\n"
        if tag == "li":
            return self._inner_text(elem).strip()

        # --- Table ---
        if tag == "table":
            return "\n\n" + self._convert_table(elem) + "\n\n"

        # --- Blockquote ---
        if tag == "blockquote":
            text = self._inner_text(elem).strip()
            return "\n\n" + "\n".join(f"> {l}" for l in text.split("\n")) + "\n\n"

        # --- Horizontal rule ---
        if tag == "hr":
            return "\n\n---\n\n"

        # --- Confluence structured macro ---
        if tag == "structured-macro":
            return self._convert_macro(elem)

        # --- Confluence image ---
        if tag == "image":
            return self._convert_ac_image(elem)

        # --- Root / generic container ---
        return self._children_text(elem)

    def _convert_macro(self, elem: ET.Element) -> str:
        macro_name = self._get_attr(elem, "name")

        if macro_name == "code":
            language = ""
            body = ""
            for child in elem:
                ltag = self._local_tag(child.tag)
                if ltag == "parameter" and self._get_attr(child, "name") == "language":
                    language = (child.text or "").strip()
                if ltag == "plain-text-body":
                    body = (child.text or "").strip()
            return f"\n\n```{language}\n{body}\n```\n\n"

        if macro_name in ("info", "note", "warning", "tip", "panel"):
            label = macro_name.capitalize()
            body = self._macro_rich_body(elem)
            return f"\n\n> **{label}:** {body}\n\n"

        if macro_name == "expand":
            title = ""
            for child in elem:
                ltag = self._local_tag(child.tag)
                if ltag == "parameter" and self._get_attr(child, "name") == "title":
                    title = (child.text or "").strip()
            body = self._macro_rich_body(elem)
            title = title or "Details"
            return f"\n\n**{title}:** {body}\n\n"

        if macro_name == "noformat":
            for child in elem:
                if self._local_tag(child.tag) == "plain-text-body":
                    return f"\n\n```\n{(child.text or '').strip()}\n```\n\n"

        # Unknown macro — try to extract body
        body = self._macro_rich_body(elem)
        return body if body else ""

    def _macro_rich_body(self, elem: ET.Element) -> str:
        for child in elem:
            if self._local_tag(child.tag) == "rich-text-body":
                return self._inner_text(child).strip()
        return ""

    def _convert_ac_image(self, elem: ET.Element) -> str:
        url = ""
        for child in elem:
            ltag = self._local_tag(child.tag)
            if ltag == "url":
                url = self._get_attr(child, "value") or child.get("href", "")
            elif ltag == "attachment":
                url = self._get_attr(child, "filename") or ""
        alt = self._get_attr(elem, "alt") or ""
        return f"![{alt}]({url})"

    def _convert_list(self, elem: ET.Element, ordered: bool) -> str:
        lines = []
        for i, child in enumerate(elem):
            if self._local_tag(child.tag) == "li":
                text = self._inner_text(child).strip()
                prefix = f"{i + 1}." if ordered else "-"
                lines.append(f"{prefix} {text}")
        return "\n".join(lines)

    def _convert_table(self, elem: ET.Element) -> str:
        rows: List[List[str]] = []

        for child in elem:
            ltag = self._local_tag(child.tag)
            if ltag == "tr":
                row = self._parse_table_row(child)
                rows.append(row)
            elif ltag in ("thead", "tbody", "tfoot"):
                for sub in child:
                    if self._local_tag(sub.tag) == "tr":
                        rows.append(self._parse_table_row(sub))

        if not rows:
            return ""

        max_cols = max(len(r) for r in rows)
        for row in rows:
            while len(row) < max_cols:
                row.append("")

        lines = []
        lines.append("| " + " | ".join(rows[0]) + " |")
        lines.append("| " + " | ".join("---" for _ in range(max_cols)) + " |")
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    def _parse_table_row(self, tr: ET.Element) -> List[str]:
        cells = []
        for cell in tr:
            ltag = self._local_tag(cell.tag)
            if ltag in ("td", "th"):
                text = self._inner_text(cell).strip().replace("\n", " ")
                cells.append(text)
        return cells

    def _inner_text(self, elem: ET.Element) -> str:
        """Recursively get text, converting child elements."""
        parts = []
        if elem.text:
            parts.append(elem.text)
        for child in elem:
            parts.append(self._walk(child))
            if child.tail:
                parts.append(child.tail)
        return "".join(parts)

    def _children_text(self, elem: ET.Element) -> str:
        """Like _inner_text but for generic containers."""
        return self._inner_text(elem)

    @staticmethod
    def _local_tag(tag: str) -> str:
        """Strip namespace from tag: ``{ns}local`` → ``local``."""
        if "}" in tag:
            return tag.split("}", 1)[1]
        return tag

    @staticmethod
    def _get_attr(elem: ET.Element, local_name: str) -> str:
        """Get attribute by local name, ignoring namespace prefix."""
        for key, val in elem.attrib.items():
            attr_local = key.split("}", 1)[-1] if "}" in key else key
            if attr_local == local_name:
                return val
        return ""

    @staticmethod
    def _strip_tags(html: str) -> str:
        """Fallback: strip all tags and return text."""
        return re.sub(r"<[^>]+>", "", html).strip()


def convert_content(body: Any, fmt: str) -> str:
    """Convert Confluence page body to Markdown.

    Args:
        body: ADF dict or XHTML string.
        fmt: ``"adf"`` for Cloud atlas_doc_format,
             ``"storage"`` or ``"xhtml"`` for On-Prem storage format.

    Returns:
        Markdown string.
    """
    if fmt == "adf":
        if isinstance(body, dict):
            return ADFConverter().convert(body)
        return ""
    if fmt in ("storage", "xhtml"):
        if isinstance(body, str):
            return XHTMLConverter().convert(body)
        return ""
    logger.warning("Unknown content format: %s", fmt)
    return str(body) if body else ""
