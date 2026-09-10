"""One-off: fix the prose defects a close read of the built article turned up.

Four of them come from the same cause. A macro that holds a small count renders as a
numeral, and a numeral reads wrong mid-sentence: "the 5 major European leagues" and
"any one of the 6 families". The fix keeps the value generated, by writing a spelled-out
companion macro from the same number rather than by typing the word into the paper.
"""

import pathlib

# --- 1. A word form for small counts, generated beside the numeral ---------------------
TABLES = pathlib.Path("src/tables.py")
text = TABLES.read_text()

OLD_ADD = '''    def add_year(self, name: str, value: Any) -> None:
        """Years must not carry a thousands separator."""
        if value is None:
            self._skipped.append(name)
            return
        self._defs[name] = str(int(value))'''
NEW_ADD = '''    def add_year(self, name: str, value: Any) -> None:
        """Years must not carry a thousands separator."""
        if value is None:
            self._skipped.append(name)
            return
        self._defs[name] = str(int(value))

    #: Spelled-out forms for the counts that appear mid-sentence. A numeral reads wrong
    #: there: "the 5 major European leagues" wants "five".
    WORDS = (
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    ).split()

    def add_word(self, name: str, value: Any) -> None:
        """Write a small count as an English word, falling back to the numeral."""
        if value is None:
            self._skipped.append(name)
            return
        n = int(value)
        self._defs[name] = self.WORDS[n] if 0 <= n < len(self.WORDS) else num(n, 0)'''
assert OLD_ADD in text, "add_year not found"
text = text.replace(OLD_ADD, NEW_ADD, 1)

OLD_COUNTS = """    m.add("ArcNSeasons", len(dig(arch, "seasons") or []))
    m.add("ArcNLeagues", len(dig(arch, "leagues") or []))"""
NEW_COUNTS = """    m.add("ArcNSeasons", len(dig(arch, "seasons") or []))
    m.add("ArcNLeagues", len(dig(arch, "leagues") or []))
    m.add_word("ArcNSeasonsWord", len(dig(arch, "seasons") or []))
    m.add_word("ArcNLeaguesWord", len(dig(arch, "leagues") or []))"""
assert OLD_COUNTS in text, "season and league counts not found"
text = text.replace(OLD_COUNTS, NEW_COUNTS, 1)

OLD_FAM = """    m.add("AblFamilies", len(dig(ab, "families") or {}))"""
NEW_FAM = """    m.add("AblFamilies", len(dig(ab, "families") or {}))
    m.add_word("AblFamiliesWord", len(dig(ab, "families") or {}))"""
assert OLD_FAM in text, "family count not found"
text = text.replace(OLD_FAM, NEW_FAM, 1)

# The noise fraction is quoted in prose as a share of the pool, so it needs a percent.
OLD_NOISE = """        m.add(f"StrNoise{tag}", hd.get("noise_fraction"), places=3)"""
NEW_NOISE = """        m.add(f"StrNoise{tag}", hd.get("noise_fraction"), places=3)
        m.add_pct(f"StrNoisePct{tag}", hd.get("noise_fraction"))"""
assert OLD_NOISE in text, "noise macro not found"
text = text.replace(OLD_NOISE, NEW_NOISE, 1)
TABLES.write_text(text)

# --- 2. The prose repairs --------------------------------------------------------------
FIXES = {
    "paper/sections/abstract.tex": [
        (
            "We test existence directly on \\ArcRows{} player-seasons from\n\\ArcNSeasons{} seasons of the big five European leagues",
            "We test existence directly on \\ArcRows{} player-seasons from\n\\ArcNSeasonsWord{} seasons of the big five European leagues",
        ),
    ],
    "paper/sections/methods.tex": [
        (
            "The analysis uses season-aggregate player statistics for the \\ArcNLeagues{} major European\nleagues over \\ArcNSeasons{} seasons from \\ArcSeasonFirst{} to \\ArcSeasonLast{}",
            "The analysis uses season-aggregate player statistics for the \\ArcNLeaguesWord{} major European\nleagues over \\ArcNSeasonsWord{} seasons from \\ArcSeasonFirst{} to \\ArcSeasonLast{}",
        ),
        (
            "outfield players above the minutes threshold of\n\\MinMinutes{} and",
            "outfield players above the eligibility threshold of\n\\MinMinutes{} minutes and",
        ),
        (
            "the same player appears in up to \\ArcNSeasons{} of them",
            "the same player appears in up to \\ArcNSeasonsWord{} of them",
        ),
    ],
    "paper/sections/introduction.tex": [
        (
            "We assemble \\ArcRows{} player-seasons covering\n\\ArcNSeasons{} seasons across the \\ArcNLeagues{} major European leagues",
            "We assemble \\ArcRows{} player-seasons covering\n\\ArcNSeasonsWord{} seasons across the \\ArcNLeaguesWord{} major European leagues",
        ),
    ],
    "paper/sections/results.tex": [
        (
            "and \\ArcPCNinety{}\nare needed for ninety percent of the variance",
            "and \\ArcPCNinety{}\nare needed to reach 90 percent of the variance",
        ),
        (
            "while midfielders divide \\ArcModeAttMF{} to \\ArcModeDefMF{}, as close to\neven as the sample permits",
            "while midfielders divide almost evenly, \\ArcSplitMF{} percent of\nthem on the attacking side",
        ),
        (
            "The errors are the argument. Of \\SupN{}\nplayer-seasons the model assigns \\SupDFasFW{} defenders to the forward class and \\SupFWasDF{}\nforwards to the defender class, while every other confusion runs through midfield, so only\n\\SupPoleShare{} percent of misclassifications skip the middle.",
            "The errors are the argument. Over \\SupN{}\nplayer-seasons, only \\SupPoleErrors{} confusions link the two extreme classes directly, against\n\\SupMiddleErrors{} that run through midfield, so \\SupPoleShare{} percent of misclassifications\nskip the middle.",
        ),
        (
            "Density-based clustering assigns \\StrNoiseAll{} of the pool to noise",
            "Density-based clustering assigns \\StrNoisePctAll{} percent of the pool to noise",
        ),
        (
            "Removing any one\nof the \\AblFamilies{} families removes nothing that matters",
            "Removing any one\nof the \\AblFamiliesWord{} families removes nothing that matters",
        ),
    ],
    "paper/sections/discussion.tex": [
        (
            "with the\nsame player appearing in up to \\ArcNSeasons{} rows",
            "with the\nsame player appearing in up to \\ArcNSeasonsWord{} rows",
        ),
    ],
}

for name, pairs in FIXES.items():
    path = pathlib.Path(name)
    body = path.read_text()
    for old, new in pairs:
        assert old in body, (name, old[:70])
        body = body.replace(old, new, 1)
    path.write_text(body)
print("ok")
