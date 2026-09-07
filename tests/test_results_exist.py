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

#: LaTeX and package commands that happen to start with a capital letter.
_UPPERCASE_BUILTINS = {
    "SI",
    "Large",
    "LaTeX",
    "TeX",
    "Huge",
    "Roman",
    "Alph",
    "AA",
    "S",
    "P",
}


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
    """Every generated macro the paper cites must exist in results/macros.tex.

    Generated macros are CamelCase by construction (``\\NEligiblePrimary``), while LaTeX
    and package commands are lowercase. Checking the capitalised namespace therefore
    catches a stale or misspelled project macro without maintaining a whitelist of every
    command the document class provides.
    """
    body = _strip_comments(_tex_bodies())
    used = set(re.findall(r"\\([A-Za-z]+)", body))
    defined = defined_macros()
    inline = set(re.findall(r"\\newcommand\s*\{?\\([A-Za-z]+)\}?", body))
    candidates = {m for m in used if m[:1].isupper()} - _UPPERCASE_BUILTINS - inline
    unknown = sorted(candidates - defined)
    assert not unknown, f"macros used by the paper but not defined in results/macros.tex: {unknown}"


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


def test_every_input_fragment_exists() -> None:
    """Every table fragment the paper inputs must have been generated.

    A missing fragment otherwise surfaces as an emergency stop deep inside the LaTeX
    run, which is a much worse way to find out than a failing test.
    """
    body = _strip_comments(_tex_bodies())
    inputs = re.findall(r"\\input\{([^}]+)\}", body)
    missing = []
    for ref in inputs:
        name = ref.strip()
        if not name.endswith(".tex"):
            name += ".tex"
        candidates = [
            config.PAPER / name,
            (config.PAPER / name).resolve(),
            config.RESULTS / name.split("/")[-1],
            config.TABLES / name.split("/")[-1],
        ]
        if not any(c.exists() for c in candidates):
            missing.append(ref)
    assert not missing, f"table fragments the paper inputs but which were not generated: {missing}"


def _macro_values() -> dict[str, str]:
    if not MACROS_FILE.exists():
        pytest.skip("run `make tables` first")
    body = MACROS_FILE.read_text(encoding="utf8", errors="ignore")
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", body))


def test_position_counts_sum_to_the_eligible_total() -> None:
    """Counts quoted as the analysis sample must actually be the analysis sample.

    The pool counts and the post-filter counts differ by more than a hundred players, and
    an earlier revision cited the pool counts alongside the eligible total, which did not
    add up. Arithmetic the reader can check has to check out.
    """
    macros = _macro_values()
    needed = ["NDFPrimary", "NMFPrimary", "NFWPrimary", "NEligiblePrimary"]
    if any(n not in macros for n in needed):
        pytest.skip("position macros not generated yet")
    total = sum(int(macros[f"N{g}Primary"].replace(",", "")) for g in ("DF", "MF", "FW"))
    assert total == int(macros["NEligiblePrimary"].replace(",", "")), (
        f"position groups sum to {total} but the eligible total is {macros['NEligiblePrimary']}"
    )


def test_pool_counts_exceed_eligible_counts() -> None:
    """Sanity check that the two count families were not swapped."""
    macros = _macro_values()
    if "NDFPool" not in macros:
        pytest.skip("pool macros not generated yet")
    for group in ("DF", "MF", "FW"):
        pool = int(macros[f"N{group}Pool"].replace(",", ""))
        eligible = int(macros[f"N{group}Primary"].replace(",", ""))
        assert pool >= eligible, f"{group}: pool {pool} is smaller than eligible {eligible}"


def _int(macros: dict[str, str], name: str) -> int:
    return int(macros[name].replace(",", ""))


def test_archive_contingency_sums_to_the_eligible_total() -> None:
    """The mode-by-position table quoted in the text must account for every player.

    The paper reads the contingency cells out loud, so a reader can add them up. They have
    to reach the eligible total, or one of the two numbers is being drawn from a different
    filter than the other.
    """
    macros = _macro_values()
    cells = [f"ArcMode{m}{g}" for m in ("Def", "Att") for g in ("DF", "MF", "FW")]
    if any(c not in macros for c in cells) or "ArcEligible" not in macros:
        pytest.skip("archive contingency macros not generated yet")
    total = sum(_int(macros, c) for c in cells)
    assert total == _int(macros, "ArcEligible"), (
        f"contingency cells sum to {total} but the eligible total is {macros['ArcEligible']}"
    )


def test_archive_pole_purity_matches_the_contingency() -> None:
    """The purity percentages must be recomputable from the counts beside them.

    The argument turns on defenders and forwards sitting at opposite poles while midfielders
    divide evenly, and both halves of that claim are quoted as percentages derived from the
    cells. If the two drift apart the sentence stops being checkable.
    """
    macros = _macro_values()
    if "ArcSplitDF" not in macros:
        pytest.skip("archive split macros not generated yet")
    for group in ("DF", "MF", "FW"):
        low = _int(macros, f"ArcModeDef{group}")
        high = _int(macros, f"ArcModeAtt{group}")
        expected = 100.0 * max(low, high) / (low + high)
        quoted = float(macros[f"ArcSplit{group}"])
        assert abs(expected - quoted) < 0.05, (
            f"{group}: cells give {expected:.1f} percent but the paper quotes {quoted}"
        )
    # The poles must actually be poles, and the middle must actually be a middle.
    assert float(macros["ArcSplitMF"]) < float(macros["ArcSplitDF"]), (
        "midfielders are more one-sided than defenders, which contradicts the argument"
    )
    assert float(macros["ArcSplitMF"]) < float(macros["ArcSplitFW"]), (
        "midfielders are more one-sided than forwards, which contradicts the argument"
    )


def test_supervised_errors_are_partitioned_correctly() -> None:
    """The pole and middle error counts must cover the off-diagonal exactly once.

    The claim that errors route through midfield is stated as a share, so the denominator
    has to be every misclassification and each cell has to be counted on one side only.
    """
    macros = _macro_values()
    needed = ["SupPoleErrors", "SupMiddleErrors", "SupPoleShare"]
    if any(n not in macros for n in needed):
        pytest.skip("supervised macros not generated yet")
    poles = _int(macros, "SupPoleErrors")
    middle = _int(macros, "SupMiddleErrors")
    assert poles == _int(macros, "SupDFasFW") + _int(macros, "SupFWasDF")
    assert middle == sum(
        _int(macros, n) for n in ("SupMFasDF", "SupMFasFW", "SupDFasMF", "SupFWasMF")
    )
    expected = 100.0 * poles / (poles + middle)
    assert abs(expected - float(macros["SupPoleShare"])) < 0.01, (
        f"pole share should be {expected:.2f} but the paper quotes {macros['SupPoleShare']}"
    )


def test_archive_elasticities_are_below_unity() -> None:
    """The possession-adjustment claim is that every fitted exponent is sub-proportional.

    The paper says so in the abstract, the results and the conclusion. If a refit ever
    produced an elasticity at or above one, those three sentences would all be wrong at
    once, so the claim is checked rather than trusted.
    """
    macros = _macro_values()
    names = [f"ArcElast{s}" for s in ("Tackles", "Interceptions", "Blocks", "Clearances", "Fouls")]
    if any(n not in macros for n in names):
        pytest.skip("archive elasticity macros not generated yet")
    values = {n: float(macros[n]) for n in names}
    over = {n: v for n, v in values.items() if v >= 1.0}
    assert not over, f"elasticities at or above one contradict the paper's claim: {over}"
    assert float(macros["ArcElastMin"]) == min(values.values())
    assert float(macros["ArcElastMax"]) == max(values.values())
