"""Tests for ADF and XHTML → Markdown content converters"""

import pytest

from src.atlassian.content_converter import ADFConverter, XHTMLConverter, convert_content


# =====================================================================
# ADF Converter Tests
# =====================================================================

class TestADFHeadings:
    def test_heading_levels(self):
        for level in range(1, 7):
            doc = {"type": "doc", "content": [
                {"type": "heading", "attrs": {"level": level}, "content": [
                    {"type": "text", "text": f"Heading {level}"}
                ]}
            ]}
            result = ADFConverter().convert(doc)
            assert result == f"{'#' * level} Heading {level}"

    def test_heading_with_inline_marks(self):
        doc = {"type": "doc", "content": [
            {"type": "heading", "attrs": {"level": 2}, "content": [
                {"type": "text", "text": "bold", "marks": [{"type": "strong"}]},
                {"type": "text", "text": " heading"},
            ]}
        ]}
        result = ADFConverter().convert(doc)
        assert result == "## **bold** heading"


class TestADFParagraph:
    def test_simple_paragraph(self):
        doc = {"type": "doc", "content": [
            {"type": "paragraph", "content": [
                {"type": "text", "text": "Hello world"}
            ]}
        ]}
        assert ADFConverter().convert(doc) == "Hello world"

    def test_multiple_paragraphs(self):
        doc = {"type": "doc", "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "First"}]},
            {"type": "paragraph", "content": [{"type": "text", "text": "Second"}]},
        ]}
        result = ADFConverter().convert(doc)
        assert "First" in result
        assert "Second" in result
        assert "\n\n" in result


class TestADFCodeBlock:
    def test_fenced_code_with_language(self):
        doc = {"type": "doc", "content": [
            {"type": "codeBlock", "attrs": {"language": "python"}, "content": [
                {"type": "text", "text": "print('hello')"}
            ]}
        ]}
        result = ADFConverter().convert(doc)
        assert result == "```python\nprint('hello')\n```"

    def test_fenced_code_no_language(self):
        doc = {"type": "doc", "content": [
            {"type": "codeBlock", "attrs": {}, "content": [
                {"type": "text", "text": "some code"}
            ]}
        ]}
        result = ADFConverter().convert(doc)
        assert result == "```\nsome code\n```"


class TestADFTable:
    def test_simple_table(self):
        doc = {"type": "doc", "content": [
            {"type": "table", "content": [
                {"type": "tableRow", "content": [
                    {"type": "tableHeader", "content": [
                        {"type": "paragraph", "content": [{"type": "text", "text": "Name"}]}
                    ]},
                    {"type": "tableHeader", "content": [
                        {"type": "paragraph", "content": [{"type": "text", "text": "Value"}]}
                    ]},
                ]},
                {"type": "tableRow", "content": [
                    {"type": "tableCell", "content": [
                        {"type": "paragraph", "content": [{"type": "text", "text": "foo"}]}
                    ]},
                    {"type": "tableCell", "content": [
                        {"type": "paragraph", "content": [{"type": "text", "text": "bar"}]}
                    ]},
                ]},
            ]}
        ]}
        result = ADFConverter().convert(doc)
        assert "| Name | Value |" in result
        assert "| --- | --- |" in result
        assert "| foo | bar |" in result


class TestADFLists:
    def test_bullet_list(self):
        doc = {"type": "doc", "content": [
            {"type": "bulletList", "content": [
                {"type": "listItem", "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": "Item A"}]}
                ]},
                {"type": "listItem", "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": "Item B"}]}
                ]},
            ]}
        ]}
        result = ADFConverter().convert(doc)
        assert "- Item A" in result
        assert "- Item B" in result

    def test_ordered_list(self):
        doc = {"type": "doc", "content": [
            {"type": "orderedList", "content": [
                {"type": "listItem", "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": "First"}]}
                ]},
                {"type": "listItem", "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": "Second"}]}
                ]},
            ]}
        ]}
        result = ADFConverter().convert(doc)
        assert "1. First" in result
        assert "2. Second" in result


class TestADFBlockElements:
    def test_blockquote(self):
        doc = {"type": "doc", "content": [
            {"type": "blockquote", "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "Quoted text"}]}
            ]}
        ]}
        result = ADFConverter().convert(doc)
        assert "> Quoted text" in result

    def test_rule(self):
        doc = {"type": "doc", "content": [{"type": "rule"}]}
        assert ADFConverter().convert(doc) == "---"

    def test_panel(self):
        doc = {"type": "doc", "content": [
            {"type": "panel", "attrs": {"panelType": "info"}, "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "Notice"}]}
            ]}
        ]}
        result = ADFConverter().convert(doc)
        assert "> **Info:** Notice" in result

    def test_expand(self):
        doc = {"type": "doc", "content": [
            {"type": "expand", "attrs": {"title": "Click me"}, "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "Hidden content"}]}
            ]}
        ]}
        result = ADFConverter().convert(doc)
        assert "**Click me:** Hidden content" in result


class TestADFInlineMarks:
    def test_strong(self):
        doc = {"type": "doc", "content": [
            {"type": "paragraph", "content": [
                {"type": "text", "text": "bold", "marks": [{"type": "strong"}]}
            ]}
        ]}
        assert "**bold**" in ADFConverter().convert(doc)

    def test_em(self):
        doc = {"type": "doc", "content": [
            {"type": "paragraph", "content": [
                {"type": "text", "text": "italic", "marks": [{"type": "em"}]}
            ]}
        ]}
        assert "*italic*" in ADFConverter().convert(doc)

    def test_code_mark(self):
        doc = {"type": "doc", "content": [
            {"type": "paragraph", "content": [
                {"type": "text", "text": "code", "marks": [{"type": "code"}]}
            ]}
        ]}
        assert "`code`" in ADFConverter().convert(doc)

    def test_link(self):
        doc = {"type": "doc", "content": [
            {"type": "paragraph", "content": [
                {"type": "text", "text": "click", "marks": [
                    {"type": "link", "attrs": {"href": "https://example.com"}}
                ]}
            ]}
        ]}
        result = ADFConverter().convert(doc)
        assert "[click](https://example.com)" in result

    def test_inline_card(self):
        doc = {"type": "doc", "content": [
            {"type": "paragraph", "content": [
                {"type": "inlineCard", "attrs": {"url": "https://example.com/page"}}
            ]}
        ]}
        result = ADFConverter().convert(doc)
        assert "[https://example.com/page](https://example.com/page)" in result

    def test_mention(self):
        doc = {"type": "doc", "content": [
            {"type": "paragraph", "content": [
                {"type": "mention", "attrs": {"text": "John Doe"}}
            ]}
        ]}
        assert "@John Doe" in ADFConverter().convert(doc)

    def test_strikethrough(self):
        doc = {"type": "doc", "content": [
            {"type": "paragraph", "content": [
                {"type": "text", "text": "deleted", "marks": [{"type": "strike"}]}
            ]}
        ]}
        assert "~~deleted~~" in ADFConverter().convert(doc)


class TestADFEdgeCases:
    def test_empty_doc(self):
        assert ADFConverter().convert({}) == ""
        assert ADFConverter().convert(None) == ""

    def test_doc_with_no_content(self):
        assert ADFConverter().convert({"type": "doc"}) == ""

    def test_unknown_node_type(self):
        doc = {"type": "doc", "content": [
            {"type": "unknownNode", "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "fallback"}]}
            ]}
        ]}
        result = ADFConverter().convert(doc)
        assert "fallback" in result


# =====================================================================
# XHTML Converter Tests
# =====================================================================

class TestXHTMLHeadings:
    def test_heading_levels(self):
        for level in range(1, 7):
            xhtml = f"<h{level}>Heading {level}</h{level}>"
            result = XHTMLConverter().convert(xhtml)
            assert f"{'#' * level} Heading {level}" in result


class TestXHTMLParagraph:
    def test_simple_paragraph(self):
        result = XHTMLConverter().convert("<p>Hello world</p>")
        assert "Hello world" in result

    def test_paragraph_with_inline(self):
        result = XHTMLConverter().convert("<p><strong>bold</strong> and <em>italic</em></p>")
        assert "**bold**" in result
        assert "*italic*" in result


class TestXHTMLCodeBlock:
    def test_code_macro_with_language(self):
        xhtml = (
            '<ac:structured-macro ac:name="code">'
            '<ac:parameter ac:name="language">python</ac:parameter>'
            '<ac:plain-text-body>print("hello")</ac:plain-text-body>'
            '</ac:structured-macro>'
        )
        result = XHTMLConverter().convert(xhtml)
        assert "```python" in result
        assert 'print("hello")' in result
        assert "```" in result

    def test_pre_tag(self):
        result = XHTMLConverter().convert("<pre>some code</pre>")
        assert "```" in result
        assert "some code" in result


class TestXHTMLTable:
    def test_simple_table(self):
        xhtml = (
            "<table>"
            "<tr><th>Name</th><th>Value</th></tr>"
            "<tr><td>foo</td><td>bar</td></tr>"
            "</table>"
        )
        result = XHTMLConverter().convert(xhtml)
        assert "| Name | Value |" in result
        assert "| --- | --- |" in result
        assert "| foo | bar |" in result

    def test_table_with_thead_tbody(self):
        xhtml = (
            "<table>"
            "<thead><tr><th>A</th><th>B</th></tr></thead>"
            "<tbody><tr><td>1</td><td>2</td></tr></tbody>"
            "</table>"
        )
        result = XHTMLConverter().convert(xhtml)
        assert "| A | B |" in result
        assert "| 1 | 2 |" in result


class TestXHTMLLists:
    def test_unordered_list(self):
        result = XHTMLConverter().convert("<ul><li>A</li><li>B</li></ul>")
        assert "- A" in result
        assert "- B" in result

    def test_ordered_list(self):
        result = XHTMLConverter().convert("<ol><li>First</li><li>Second</li></ol>")
        assert "1. First" in result
        assert "2. Second" in result


class TestXHTMLLinks:
    def test_anchor_tag(self):
        result = XHTMLConverter().convert('<a href="https://example.com">click</a>')
        assert "[click](https://example.com)" in result


class TestXHTMLMacros:
    def test_info_panel(self):
        xhtml = (
            '<ac:structured-macro ac:name="info">'
            '<ac:rich-text-body><p>Important note</p></ac:rich-text-body>'
            '</ac:structured-macro>'
        )
        result = XHTMLConverter().convert(xhtml)
        assert "> **Info:**" in result
        assert "Important note" in result

    def test_warning_panel(self):
        xhtml = (
            '<ac:structured-macro ac:name="warning">'
            '<ac:rich-text-body><p>Be careful</p></ac:rich-text-body>'
            '</ac:structured-macro>'
        )
        result = XHTMLConverter().convert(xhtml)
        assert "> **Warning:**" in result

    def test_expand_macro(self):
        xhtml = (
            '<ac:structured-macro ac:name="expand">'
            '<ac:parameter ac:name="title">Details</ac:parameter>'
            '<ac:rich-text-body><p>Expanded content</p></ac:rich-text-body>'
            '</ac:structured-macro>'
        )
        result = XHTMLConverter().convert(xhtml)
        assert "**Details:**" in result
        assert "Expanded content" in result

    def test_noformat_macro(self):
        xhtml = (
            '<ac:structured-macro ac:name="noformat">'
            '<ac:plain-text-body>raw text here</ac:plain-text-body>'
            '</ac:structured-macro>'
        )
        result = XHTMLConverter().convert(xhtml)
        assert "```" in result
        assert "raw text here" in result


class TestXHTMLImage:
    def test_img_tag(self):
        result = XHTMLConverter().convert('<img src="photo.png" alt="Photo" />')
        assert "![Photo](photo.png)" in result

    def test_ac_image(self):
        xhtml = (
            '<ac:image>'
            '<ri:attachment ri:filename="diagram.png" />'
            '</ac:image>'
        )
        result = XHTMLConverter().convert(xhtml)
        assert "![](diagram.png)" in result


class TestXHTMLBlockElements:
    def test_blockquote(self):
        result = XHTMLConverter().convert("<blockquote>Quoted text</blockquote>")
        assert "> Quoted text" in result

    def test_horizontal_rule(self):
        result = XHTMLConverter().convert("<hr />")
        assert "---" in result

    def test_inline_code(self):
        result = XHTMLConverter().convert("<code>foo()</code>")
        assert "`foo()`" in result


class TestXHTMLEdgeCases:
    def test_empty_input(self):
        assert XHTMLConverter().convert("") == ""
        assert XHTMLConverter().convert("   ") == ""

    def test_plain_text(self):
        result = XHTMLConverter().convert("Just plain text")
        assert "Just plain text" in result


# =====================================================================
# convert_content() dispatch tests
# =====================================================================

class TestConvertContent:
    def test_adf_dispatch(self):
        doc = {"type": "doc", "content": [
            {"type": "paragraph", "content": [{"type": "text", "text": "ADF text"}]}
        ]}
        assert "ADF text" in convert_content(doc, "adf")

    def test_storage_dispatch(self):
        assert "Storage text" in convert_content("<p>Storage text</p>", "storage")

    def test_xhtml_dispatch(self):
        assert "XHTML text" in convert_content("<p>XHTML text</p>", "xhtml")

    def test_adf_with_non_dict(self):
        assert convert_content("not a dict", "adf") == ""

    def test_storage_with_non_string(self):
        assert convert_content({"not": "string"}, "storage") == ""

    def test_unknown_format(self):
        result = convert_content("some body", "unknown_fmt")
        assert "some body" in result
