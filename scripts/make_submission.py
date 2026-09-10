"""Assemble a self-contained arXiv submission from the working tree.

The repository layout is convenient to build from and wrong to upload. `paper/figures`
is a symlink, which does not survive a tarball, and `main.tex` reaches outside its own
directory for the generated macros and tables, which arXiv will not resolve. This script
flattens both, ships only the figures the document actually includes, pins the date so
the submitted PDF is reproducible, and then compiles the staged copy from scratch to
prove it stands on its own.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
STAGE = ROOT / "submission"
ARCHIVE = ROOT / "submission.tar.gz"


def read(path: Path) -> str:
    return path.read_text(encoding="utf8")


def collect_included_figures(sources: list[Path]) -> set[str]:
    """Only the figures the document includes, so nothing dead is uploaded."""
    names: set[str] = set()
    for src in sources:
        for match in re.findall(r"\\includegraphics[^{]*\{([^}]+)\}", read(src)):
            names.add(Path(match).name)
    return names


def rewrite_main(text: str, date: str) -> str:
    """Point every path inside the staging directory and pin the date."""
    text = text.replace(r"\input{../results/macros.tex}", r"\input{macros.tex}")
    text = re.sub(r"\\input\{\.\./results/tables/([^}]+)\}", r"\\input{tables/\1}", text)
    # \today would make the submitted PDF differ from the one checked here.
    text = text.replace(r"\date{\today}", f"\\date{{{date}}}")
    return text


def rewrite_section(text: str) -> str:
    return re.sub(r"\\input\{\.\./results/tables/([^}]+)\}", r"\\input{tables/\1}", text)


def build(stage: Path) -> None:
    """Compile the staged copy, failing loudly on anything arXiv would also reject."""
    env_path = f"{Path.home()}/Library/TinyTeX/bin/universal-darwin"
    env = {
        **dict(__import__("os").environ),
        "PATH": f"{env_path}:{__import__('os').environ['PATH']}",
    }
    # The article and the supplement reference each other through xr, so each needs the
    # other's .aux file and the pair is compiled twice in sequence.
    for _ in range(2):
        for name in ("supplementary.tex", "main.tex"):
            result = subprocess.run(
                ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", name],
                cwd=stage,
                env=env,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                sys.stderr.write(result.stdout[-3000:])
                raise SystemExit(f"the staged submission does not compile: {name}")

    for stem in ("main", "supplementary"):
        log = (stage / f"{stem}.log").read_text(encoding="utf8", errors="ignore")
        for pattern, message in (
            (r"LaTeX Warning: Citation .* undefined", "undefined citation"),
            (r"LaTeX Warning: Reference .* undefined", "undefined reference"),
            (r"multiply-defined", "multiply defined label"),
        ):
            if re.search(pattern, log):
                raise SystemExit(f"the staged {stem} has an {message}")
        pages = re.findall(rf"Output written on {stem}\.pdf \((\d+) pages", log)
        print(f"  staged {stem} compiles cleanly: {pages[-1] if pages else '?'} pages")


def render_abstract(macros: str, abstract: str) -> str:
    """Expand the abstract into plain text for the arXiv metadata form.

    arXiv wants the abstract typed into a web form, not read out of the source, so it is
    the one place a number could be transcribed by hand and drift. Rendering it from the
    same macro file the PDF uses removes that risk.
    """
    values = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", macros))
    text = re.sub(r"(?<!\\)%.*", "", abstract)

    def expand(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise SystemExit(f"abstract cites an undefined macro: {name}")
        return values[name]

    text = re.sub(r"\\([A-Za-z]+)\{\}", expand, text)
    text = re.sub(r"\\cite[tp]?\{([^}]*)\}", r"[\1]", text)
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    # arXiv's abstract field is plain text: math mode, thin spaces and ties have to go.
    text = text.replace("\\,", " ").replace("\\%", "%")
    text = text.replace("$", "").replace("~", " ")
    # Math mode spaces relations for us; plain text does not.
    text = re.sub(r"([<>=])(?=\d)", r"\1 ", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("\\", "")
    return " ".join(text.split())


def write_metadata(stage: Path, date: str) -> None:
    """Everything the arXiv form asks for, in one file, so nothing is retyped."""
    main = read(PAPER / "main.tex")
    title = re.search(r"\\title\{(.+?)\}\n", main, re.S)
    title_text = " ".join(title.group(1).split()) if title else "(no title found)"
    authors = re.search(r"\\author\{(.+?)\}\n", main, re.S)
    # Drop the corresponding-author footnote; the form wants names only.
    author_text = re.sub(r"\\thanks\{[^}]*\}", "", authors.group(1)) if authors else ""
    author_text = author_text.replace(r"\and", "|").strip()
    author_text = ", ".join(part.strip() for part in author_text.split("|") if part.strip())

    abstract = render_abstract(
        read(RESULTS / "macros.tex"), read(PAPER / "sections" / "abstract.tex")
    )
    log = read(stage / "main.log")
    pages = re.findall(r"Output written on main\.pdf \((\d+) pages", log)
    figures = len(list((stage / "figures").glob("*")))

    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    repo = re.sub(r"^git@github\.com:", "https://github.com/", remote).removesuffix(".git")

    body = f"""# arXiv submission

Generated by `make submission`. Every field below is read from the built paper, so it
cannot drift from what compiles.

## Upload

`submission.tar.gz` at the repository root. It unpacks flat and was compiled from a bare
directory as a check, so it does not depend on anything in this repository.

## Title

{title_text}

## Authors

{author_text}

## Abstract

{abstract}

## Comments field

{pages[-1] if pages else "?"} pages, {figures} figures. Code and data: {repo}

## Categories

Primary: stat.AP (Applications). The paper contributes a finding, evidenced entirely
within one domain's data, and proposes no method. arXiv's stat.AP remit covers the social
sciences, which is where sports analytics sits.

Cross-list: cs.LG (Machine Learning) and stat.ML, which is where the cluster validity
argument finds its other audience. cs.LG is defensible as primary instead, since its scope
explicitly admits applications of machine learning methods, and it reaches far more
readers. The trade is audience shape rather than correctness: stat.AP is smaller but is
read by the people who publish the role taxonomies this paper argues against.

Do not list stat.ME. Its scope names Model Selection and Multivariate Methods, so choosing
the number of clusters looks like a fit, but stat.ME is for papers that contribute
methodology and this one applies an existing check. The null calibration is the gap
statistic's reference-distribution idea, which the paper cites rather than claims.

## Optional classification fields

Both are optional on the submission form and both are worth filling in, because they are
what indexers and subject bibliographies key on.

MSC class: 62H30, 62P25
  62H30 is classification and discrimination, cluster analysis. 62P25 is applications of
  statistics to the social sciences.

ACM class: I.5.3
  Pattern Recognition, Clustering.

## What only you can do

arXiv gates the three steps below on your identity, so they cannot be automated.

1. **Register**, at https://arxiv.org/user/register, with your university address rather
   than a personal one. arXiv treats an institutional address as evidence of community
   membership and it is the difference between a fast path and a slow one.
2. **Get endorsed.** arXiv requires endorsement before a first submission in a category.
   Start the submission and pick the category; arXiv emails you a six-character code to
   pass to an endorser. The natural endorsers are authors of the arXiv papers this work
   cites, because an endorser should know the subject area:
   - Aalbers and Van Haaren, arXiv:1809.05173, cross-listed in exactly our three
     categories and the closest paper to ours in subject.
   - Pappalardo et al., arXiv:1802.04987, PlayeRank.
   - McInnes et al., arXiv:1802.03426, UMAP.
   - Lundberg and Lee, arXiv:1705.07874, SHAP.
   Each abstract page has a "Which authors of this paper are endorsers?" link, visible
   once you are logged in, which tells you who currently qualifies. Endorsers must have
   published in the domain within the last five years, so check that link rather than
   assuming. Do not mass-email candidates; arXiv considers that abuse.
3. **Submit.** Upload `submission.tar.gz`, paste the fields above, choose a licence, and
   accept the terms yourself. Expect a moderation hold of anywhere from a day to a week.

## Google Scholar

There is nothing to submit. Scholar crawls arXiv and will index the paper on its own,
typically within days to a few weeks of it going live. Two things are worth doing once it
appears:

- Create a Scholar profile at https://scholar.google.com/citations, sign in with your
  Google account, and claim the paper. This is what makes you searchable as an author
  rather than just the paper being findable.
- Link the profile to your ORCID iD, which keeps authorship attached to you rather than
  to a name string that another Zachary Chung could collide with.

## Date of this package

{date}
"""
    (ROOT / "SUBMISSION.md").write_text(body, encoding="utf8")
    print(f"  wrote SUBMISSION.md ({pages[-1] if pages else '?'} pages, {figures} figures)")


def main() -> None:
    date = (
        subprocess.run(
            ["git", "log", "-1", "--format=%cd", "--date=format:%B %-d, %Y"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
        or "September 6, 2026"
    )

    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "sections").mkdir(parents=True)
    (STAGE / "tables").mkdir()
    (STAGE / "figures").mkdir()

    sections = sorted((PAPER / "sections").glob("*.tex"))
    for src in sections:
        (STAGE / "sections" / src.name).write_text(rewrite_section(read(src)), encoding="utf8")

    for driver in ("main.tex", "supplementary.tex", "preamble.tex"):
        (STAGE / driver).write_text(rewrite_main(read(PAPER / driver), date), encoding="utf8")
    shutil.copy2(PAPER / "references.bib", STAGE / "references.bib")
    shutil.copy2(RESULTS / "macros.tex", STAGE / "macros.tex")

    # arXiv runs bibtex, but shipping the .bbl removes a class of failure entirely.
    for stem in ("main", "supplementary"):
        bbl = PAPER / f"{stem}.bbl"
        if not bbl.exists():
            raise SystemExit(f"paper/{stem}.bbl is missing; run `make paper` first")
        shutil.copy2(bbl, STAGE / f"{stem}.bbl")

    used_tables = set()
    drivers = [STAGE / "main.tex", STAGE / "supplementary.tex"]
    for src in [*drivers, *sorted((STAGE / "sections").glob("*.tex"))]:
        used_tables.update(re.findall(r"\\input\{tables/([^}]+)\}", read(src)))
    for name in sorted(used_tables):
        shutil.copy2(RESULTS / "tables" / name, STAGE / "tables" / name)

    wanted = collect_included_figures(sections)
    missing = [n for n in sorted(wanted) if not (FIGURES / n).exists()]
    if missing:
        raise SystemExit(f"figures referenced but not generated: {missing}")
    for name in sorted(wanted):
        shutil.copy2(FIGURES / name, STAGE / "figures" / name)

    print(f"staged {len(sections)} sections, {len(used_tables)} tables, {len(wanted)} figures")
    build(STAGE)
    write_metadata(STAGE, date)

    # arXiv wants sources, not the build products of our own test compile.
    for junk in STAGE.glob("main.*"):
        if junk.suffix not in {".tex", ".bbl", ".pdf"}:
            junk.unlink()

    with tarfile.open(ARCHIVE, "w:gz") as tar:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file() and path.name != "main.pdf":
                tar.add(path, arcname=str(path.relative_to(STAGE)))

    size = ARCHIVE.stat().st_size / 1_048_576
    print(f"wrote {ARCHIVE.relative_to(ROOT)} ({size:.1f} MB)")
    print(f"proof PDF at {(STAGE / 'main.pdf').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
