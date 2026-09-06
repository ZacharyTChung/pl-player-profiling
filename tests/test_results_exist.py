"""Verify that everything the paper references was actually generated.

The paper is forbidden from containing hard-coded numbers, so every quantity is cited
through a macro defined in ``results/macros.tex`` and every figure is included from
``figures/``. This module parses the LaTeX sources and asserts that each referenced
macro and each referenced graphic exists, which turns a silently wrong paper into a
failing test.
"""

from __future__ import annotations

import re

import pytest

import config

TEX_FILES = sorted(config.PAPER.rglob("*.tex"))
MACROS_FILE = config.RESULTS / "macros.tex"

#: Macros provided by LaTeX itself or by the packages the paper loads.
_BUILTIN_PREFIXES = (
    "begin",
    "end",
    "section",
    "subsection",
    "subsubsection",
    "paragraph",
    "cite",
    "citep",
    "citet",
    "citealp",
    "ref",
    "label",
    "input",
    "include",
    "includegraphics",
    "caption",
    "textbf",
    "textit",
    "emph",
    "texttt",
    "textsc",
    "footnote",
    "item",
    "usepackage",
    "documentclass",
    "title",
    "author",
    "date",
    "maketitle",
    "newcommand",
    "renewcommand",
    "bibliography",
    "bibliographystyle",
    "toprule",
    "midrule",
    "bottomrule",
    "cmidrule",
    "multicolumn",
    "multirow",
    "hline",
    "centering",
    "small",
    "footnotesize",
    "large",
    "Large",
    "num",
    "si",
    "SI",
    "percent",
    "приложение",
    "appendix",
    "clearpage",
    "newpage",
    "noindent",
    "quad",
    "qquad",
    "hspace",
    "vspace",
    "left",
    "right",
    "frac",
    "times",
    "leq",
    "geq",
    "alpha",
    "beta",
    "sigma",
    "mu",
    "subfloat",
    "subfigure",
    "url",
    "href",
    "today",
    "tableofcontents",
    "abstract",
    "maketitleabstract",
    "phantomsection",
    "addcontentsline",
    "text",
    "mathrm",
    "hat",
    "bar",
    "sum",
    "log",
    "exp",
    "sqrt",
    "cdot",
    "ldots",
    "dots",
    "textwidth",
    "linewidth",
    "columnwidth",
    "resizebox",
    "scriptsize",
    "tiny",
    "normalsize",
    "raggedright",
    "arraybackslash",
    "toprule",
    "addlinespace",
    "bfseries",
    "itshape",
)


def _tex_bodies() -> str:
    if not TEX_FILES:
        pytest.skip("no LaTeX sources yet")
    return "\n".join(p.read_text(encoding="utf8", errors="ignore") for p in TEX_FILES)


def _strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def defined_macros() -> set[str]:
    if not MACROS_FILE.exists():
        return set()
    body = _strip_comments(MACROS_FILE.read_text(encoding="utf8", errors="ignore"))
    return set(re.findall(r"\\(?:newcommand|providecommand)\s*\{?\\([A-Za-z]+)\}?", body))


def test_macros_file_exists() -> None:
    if not TEX_FILES:
        pytest.skip("no LaTeX sources yet")
    assert MACROS_FILE.exists(), "results/macros.tex is missing; run `make tables`"


def test_every_included_graphic_exists() -> None:
    body = _strip_comments(_tex_bodies())
    refs = re.findall(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", body)
    if not refs:
        pytest.skip("paper includes no graphics yet")

    missing = []
    for ref in refs:
        name = ref.strip()
        candidates = [
            config.PAPER / name,
            config.FIGURES / name,
            config.FIGURES / f"{name}.pdf",
            config.FIGURES / f"{name}.png",
            config.PAPER / f"{name}.pdf",
        ]
        # Strip a leading figures/ prefix, which resolves through the symlink.
        stem = name.split("/")[-1]
        candidates += [config.FIGURES / stem, config.FIGURES / f"{stem}.pdf"]
        if not any(c.exists() for c in candidates):
            missing.append(ref)
    assert not missing, f"figures referenced by the paper but not generated: {missing}"


def test_every_cited_macro_is_defined() -> None:
    body = _strip_comments(_tex_bodies())
    used = set(re.findall(r"\\([A-Za-z]+)", body))
    defined = defined_macros()
    unknown = {
        m
        for m in used
        if m not in defined and not m.startswith(_BUILTIN_PREFIXES) and m not in _BUILTIN_PREFIXES
    }
    # Macros the paper defines inline in its own preamble are fine.
    inline = set(re.findall(r"\\newcommand\s*\{?\\([A-Za-z]+)\}?", body))
    unknown -= inline
    assert not unknown, (
        f"macros used by the paper but not defined in results/macros.tex: {sorted(unknown)}"
    )


def test_no_hardcoded_numbers_in_result_sentences() -> None:
    """Results prose must cite macros, not literal figures.

    Numbers are legitimate in tables, captions with figure counts, hyperparameters and
    section numbering, so this only inspects the results sections' body prose.
    """
    results = [p for p in TEX_FILES if "result" in p.name.lower()]
    if not results:
        pytest.skip("no results section yet")
    offenders = []
    for path in results:
        body = _strip_comments(path.read_text(encoding="utf8", errors="ignore"))
        body = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "", body, flags=re.S)
        body = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", body, flags=re.S)
        body = re.sub(r"\\(?:label|ref|cite[a-z]*)\{[^}]*\}", "", body)
        for match in re.finditer(r"(?<![\w\\{])\d+\.\d+(?![\w}])", body):
            offenders.append(f"{path.name}: {match.group(0)}")
    assert not offenders, "decimal literals in results prose; cite a macro instead: " + ", ".join(
        offenders[:10]
    )


def test_all_generated_figures_are_referenced() -> None:
    """A figure nobody cites is dead weight and usually signals a forgotten reference."""
    if not TEX_FILES:
        pytest.skip("no LaTeX sources yet")
    body = _strip_comments(_tex_bodies())
    refs = {
        r.split("/")[-1].rsplit(".", 1)[0]
        for r in re.findall(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}", body)
    }
    if not refs:
        pytest.skip("paper includes no graphics yet")
    generated = {p.stem for p in config.FIGURES.glob("*.pdf")}
    orphans = sorted(generated - refs)
    assert not orphans, f"figures generated but never referenced by the paper: {orphans}"
