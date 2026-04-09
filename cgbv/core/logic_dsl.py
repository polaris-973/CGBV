from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SortDecl:
    name: str
    type: str
    values: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FunctionDecl:
    name: str
    domain: list[str]
    range: str


@dataclass(slots=True)
class VariableDecl:
    name: str
    sort: str


@dataclass(slots=True)
class SentenceDSL:
    nl: str
    logic: Any


@dataclass(slots=True)
class TheoryDSL:
    sorts: list[SortDecl] = field(default_factory=list)
    functions: list[FunctionDecl] = field(default_factory=list)
    constants: dict[str, Any] = field(default_factory=dict)
    variables: list[VariableDecl] = field(default_factory=list)
    background_constraints: list[Any] = field(default_factory=list)
    sentences: list[SentenceDSL] = field(default_factory=list)
    query: dict[str, Any] = field(default_factory=dict)
