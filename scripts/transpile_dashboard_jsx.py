#!/usr/bin/env python3
"""Transpile the dashboard React JSX file to plain React.createElement calls.

This script provides a minimal JSX transformer tailored for the Memoria dashboard
source. It removes JSX syntax so the output can run in environments that do not
support JSX parsing.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass


@dataclass
class JSXAttribute:
    name: str
    value: str
    kind: str  # "string" | "expression"


@dataclass
class JSXText:
    value: str


@dataclass
class JSXExpression:
    code: str


@dataclass
class JSXElement:
    name: str
    attributes: list[JSXAttribute]
    children: list[object]


class JSXTransformer:
    def __init__(self, source: str):
        self.source = source
        self.length = len(source)
        self.pos = 0
        self.output: list[str] = []

    def transform(self) -> str:
        while self.pos < self.length:
            ch = self.source[self.pos]
            if ch == "<" and self._is_jsx_start(self.pos):
                element = self._parse_element()
                self.output.append(self._generate_element(element))
            else:
                self.output.append(ch)
                self.pos += 1
        return "".join(self.output)

    # ------------------------------------------------------------------ parsing
    def _is_jsx_start(self, index: int) -> bool:
        if self.source[index] != "<":
            return False
        if index + 1 >= self.length:
            return False
        next_char = self.source[index + 1]
        if next_char in {"/", "!", "?"}:
            return False
        if not (next_char.isalpha() or next_char in {"_", "$"}):
            return False
        j = index - 1
        while j >= 0 and self.source[j] in " \t\r\n":
            j -= 1
        if j < 0:
            return True
        prev_char = self.source[j]
        if prev_char in "([=,{:?;!&|%^+-*/~<>":
            return True
        if prev_char.isalpha() or prev_char in {"_", "$"}:
            k = j
            while k >= 0 and (self.source[k].isalnum() or self.source[k] in {"_", "$"}):
                k -= 1
            word = self.source[k + 1 : j + 1]
            if word == "return":
                return True
        return False

    def _parse_element(self) -> JSXElement:
        assert self.source[self.pos] == "<"
        self.pos += 1
        name = self._parse_tag_name()
        attributes: list[JSXAttribute] = []
        self._skip_whitespace()
        self_closing = False
        while self.pos < self.length:
            if self._match("/>"):
                self_closing = True
                self.pos += 2
                break
            if self._peek() == ">":
                self.pos += 1
                break
            attribute = self._parse_attribute()
            attributes.append(attribute)
            self._skip_whitespace()
        children: list[object] = []
        if not self_closing:
            while self.pos < self.length:
                prev_pos = self.pos
                text = self._parse_text()
                if text:
                    children.append(JSXText(text))
                if self._match("</"):
                    self.pos += 2
                    closing_name = self._parse_tag_name()
                    if closing_name != name:
                        # Allow mismatch but ensure closing token is consumed
                        pass
                    self._skip_whitespace()
                    if self._peek() != ">":
                        raise ValueError("Expected closing '>' for </%s>" % closing_name)
                    self.pos += 1
                    break
                if self.pos >= self.length:
                    break
                ch = self._peek()
                if ch == "<":
                    child = self._parse_element()
                    children.append(child)
                elif ch == "{":
                    expr = self._parse_braced_expression()
                    children.append(JSXExpression(expr))
                else:
                    # Unhandled character, treat as raw text to avoid infinite loop
                    fallback = self._consume_until({"<", "{"})
                    if fallback:
                        children.append(JSXText(fallback))
                if self.pos == prev_pos:
                    raise RuntimeError(
                        f"Parser stalled at position {self.pos}: "
                        f"{self.source[self.pos:self.pos + 40]!r}"
                    )
        return JSXElement(name=name, attributes=attributes, children=children)

    def _parse_tag_name(self) -> str:
        start = self.pos
        while self.pos < self.length and (
            self.source[self.pos].isalnum()
            or self.source[self.pos] in {"_", "$", ".", ":"}
        ):
            self.pos += 1
        if start == self.pos:
            raise ValueError("Expected JSX tag name at position %d" % start)
        return self.source[start:self.pos]

    def _parse_attribute(self) -> JSXAttribute:
        name = self._parse_attribute_name()
        self._skip_whitespace()
        if self._peek() == "=":
            self.pos += 1
            self._skip_whitespace()
            ch = self._peek()
            if ch in {'"', "'"}:
                value = self._parse_quoted_string(ch)
                return JSXAttribute(name=name, value=value, kind="string")
            if ch == "{":
                expr = self._parse_braced_expression()
                expr = transform_jsx(expr)
                return JSXAttribute(name=name, value=expr.strip(), kind="expression")
            value = self._parse_unquoted()
            return JSXAttribute(name=name, value=value, kind="expression")
        return JSXAttribute(name=name, value="true", kind="expression")

    def _parse_attribute_name(self) -> str:
        start = self.pos
        while self.pos < self.length and (
            self.source[self.pos].isalnum()
            or self.source[self.pos] in {"_", "$", "-", ":"}
        ):
            self.pos += 1
        if start == self.pos:
            raise ValueError("Expected attribute name at position %d" % start)
        return self.source[start:self.pos]

    def _parse_quoted_string(self, quote: str) -> str:
        assert self._peek() == quote
        self.pos += 1
        chars: list[str] = []
        while self.pos < self.length:
            ch = self.source[self.pos]
            if ch == "\\":
                if self.pos + 1 < self.length:
                    chars.append(self.source[self.pos : self.pos + 2])
                    self.pos += 2
                else:
                    self.pos += 1
            elif ch == quote:
                self.pos += 1
                break
            else:
                chars.append(ch)
                self.pos += 1
        return "".join(chars)

    def _parse_unquoted(self) -> str:
        start = self.pos
        while self.pos < self.length and self.source[self.pos] not in " \t\r\n>":
            self.pos += 1
        return self.source[start:self.pos]

    def _parse_text(self) -> str:
        start = self.pos
        while self.pos < self.length and self.source[self.pos] not in "<{":
            self.pos += 1
        return self.source[start:self.pos]

    def _parse_braced_expression(self) -> str:
        assert self._peek() == "{"
        self.pos += 1
        depth = 1
        start = self.pos
        while self.pos < self.length:
            ch = self.source[self.pos]
            if ch == "{":
                depth += 1
                self.pos += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    end = self.pos
                    self.pos += 1
                    return self.source[start:end]
                self.pos += 1
                continue
            if ch in {'"', "'"}:
                self.pos = self._skip_string(self.pos)
                continue
            if ch == "`":
                self.pos = self._skip_template(self.pos)
                continue
            if ch == "/" and self.pos + 1 < self.length:
                nxt = self.source[self.pos + 1]
                if nxt == "/":
                    self.pos += 2
                    while self.pos < self.length and self.source[self.pos] not in "\r\n":
                        self.pos += 1
                    continue
                if nxt == "*":
                    self.pos += 2
                    while self.pos + 1 < self.length and not (
                        self.source[self.pos] == "*" and self.source[self.pos + 1] == "/"
                    ):
                        self.pos += 1
                    self.pos += 2
                    continue
            self.pos += 1
        raise ValueError("Unterminated JSX expression starting at %d" % start)

    def _skip_string(self, index: int) -> int:
        quote = self.source[index]
        i = index + 1
        while i < self.length:
            ch = self.source[i]
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                return i + 1
            i += 1
        return i

    def _skip_template(self, index: int) -> int:
        i = index + 1
        while i < self.length:
            ch = self.source[i]
            if ch == "\\":
                i += 2
                continue
            if ch == "`":
                return i + 1
            if ch == "$" and i + 1 < self.length and self.source[i + 1] == "{":
                i += 2
                depth = 1
                while i < self.length and depth > 0:
                    ch2 = self.source[i]
                    if ch2 == "{":
                        depth += 1
                    elif ch2 == "}":
                        depth -= 1
                    elif ch2 in {'"', "'"}:
                        i = self._skip_string(i)
                        continue
                    elif ch2 == "`":
                        i = self._skip_template(i)
                        continue
                    i += 1
                continue
            i += 1
        return i

    def _skip_whitespace(self) -> None:
        while self.pos < self.length and self.source[self.pos] in " \t\r\n":
            self.pos += 1

    def _match(self, token: str) -> bool:
        return self.source.startswith(token, self.pos)

    def _peek(self) -> str:
        if self.pos >= self.length:
            return ""
        return self.source[self.pos]

    def _consume_until(self, stop_chars: set[str]) -> str:
        start = self.pos
        while self.pos < self.length and self.source[self.pos] not in stop_chars:
            self.pos += 1
        return self.source[start:self.pos]

    # ----------------------------------------------------------- code generation
    def _generate_element(self, node: JSXElement) -> str:
        tag_code = self._format_tag_name(node.name)
        props_code = self._format_props(node.attributes)
        children_code: list[str] = []
        for child in node.children:
            if isinstance(child, JSXText):
                text = self._normalise_text(child.value)
                if text == "":
                    continue
                children_code.append(json.dumps(text))
            elif isinstance(child, JSXExpression):
                expr = transform_jsx(child.code)
                expr = expr.strip()
                if expr == "":
                    continue
                children_code.append(expr)
            elif isinstance(child, JSXElement):
                children_code.append(self._generate_element(child))
        if props_code is None:
            props_code = "null"
        args = [tag_code, props_code]
        args.extend(children_code)
        return f"React.createElement({', '.join(args)})"

    def _format_tag_name(self, name: str) -> str:
        if name and name[0].islower() and "." not in name and ":" not in name:
            return json.dumps(name)
        return name

    def _format_props(self, attributes: list[JSXAttribute]) -> str | None:
        if not attributes:
            return None
        parts = []
        for attr in attributes:
            key = json.dumps(attr.name)
            if attr.kind == "string":
                value_code = json.dumps(attr.value)
            else:
                value_code = attr.value
            parts.append(f"{key}: {value_code}")
        return "{ " + ", ".join(parts) + " }"

    def _normalise_text(self, text: str) -> str:
        if text.strip() == "":
            return ""
        text = text.replace("\r\n", "\n")
        if "\n" in text:
            text = text.replace("\n", " ")
            text = text.replace("\t", " ")
            text = re.sub(r" {2,}", " ", text)
            text = text.strip()
        else:
            text = text.replace("\t", " ")
            text = re.sub(r" {2,}", " ", text)
        return text


def transform_jsx(source: str) -> str:
    transformer = JSXTransformer(source)
    return transformer.transform()


def main() -> None:
    parser = argparse.ArgumentParser(description="Transpile dashboard/app.js JSX")
    parser.add_argument(
        "--input",
        default="dashboard/app.js",
        help="Path to the JSX source file (default: dashboard/app.js)",
    )
    parser.add_argument(
        "--output",
        default="dashboard/app.bundle.js",
        help="Where to write the transpiled output (default: dashboard/app.bundle.js)",
    )
    args = parser.parse_args()
    with open(args.input, "r", encoding="utf-8") as fh:
        source = fh.read()
    transpiled = transform_jsx(source)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(transpiled)


if __name__ == "__main__":
    main()
