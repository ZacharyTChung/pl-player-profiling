# Response to the review

Every comment from the review is below with what changed. The paper is now two documents:
`paper/main.pdf` is the article, `paper/supplementary.pdf` is the supporting information.
`make review` builds a third, the double-spaced de-identified manuscript a journal wants at
submission.

| | reviewed version | now |
|---|---|---|
| Article | 93 pages, one file | **21 pages** including references |
| Supporting information | inside the same file | separate 62-page document |
| Article word count | 18,400 | **8,700** |
| Main-text figures | 41 | **6**, each with lettered panels |

---

## Abstract

**"First sentence is way too technical, the partitioning algorithm portion comes out of nowhere."**
It now opens on the football: analysts sort players into archetypes such as the ball-playing
centre back. The algorithm problem arrives in the second sentence, as the reason the question
has not been answered.

**"The two-mode division comes out of nowhere, needs a precedent to set that up."**
The abstract now names what the two modes are, a defensive and an attacking mode of
involvement, before claiming they are real.

**"Too many numbers."**
Twelve figures down to four: the sample, the margin against the hardest null, the bootstrap
index clusterless data reach, and the agreement with listed position. It is 225 words, against
the roughly 200 that the target journals ask for.

## Contents

**"You dont really need a table of contents."** Removed from the article. The supplement keeps
one, because it is a 62-page reference document.

## Introduction

**"Put all citations at the end of the sentence and do [1-3]."**
Citations are numeric and compressed throughout. A run of three prints as `[4-7]`.

**"The idea of partitions needs to be introduced and explained."**
A paragraph now explains what a clustering algorithm does before the argument depends on it:
it is handed points and a number of groups, it divides them, and nothing in that procedure
asks whether the points were grouped to begin with.

**"The text reads funny, only an AI would know it."**
The sentence you quoted now reads "Getting clusters out of a clustering algorithm is therefore
not evidence that clusters were there to begin with." Every section was rewritten in one pass,
which is the fix for the inconsistent phrasing you noticed.

**"The introduction is really really long, you dont need all 6 results in that detail."**
2,040 words to 1,130, following the template you gave: context, the problem, what the
literature supports, what we do, what we expect. The six numbered findings are gone; what the
paper finds is one paragraph.

## Related work

**"This section would go into the introduction. You can heavily cut it."**
Deleted as a section. It is now two paragraphs of the introduction, clustered as you suggested:
work on proprietary feeds, the public-aggregate taxonomy genre, the cluster-validity literature,
and the closest relative. 1,100 words to about 450.

## Data

**"You can merge into the methods."** It is now the first subsection of Methods.

## Methods

**"It's good, but it's really really dense. Heavily cut it, or put it at the end."**
Both, in the proportion you preferred. 2,910 words to 1,889 in the article; the procedural half,
the generators stated formally, the stability and replication machinery, the ablation protocol,
the outcome audit, profiling and attribution, is Section S1 of the supplement. A reader
following the argument no longer steps over it; a reader reproducing the work has it in full.

## Results

**"5-6 figures max, merged into larger figures that tell a story."**
Six, each carrying one argument across lettered panels:

1. Two modes are real and nothing finer is (six panels)
2. The shape of the space, and what the modes are not (six panels)
3. What the two modes are (three panels)
4. It reproduces in every season, league and feature family (eight panels)
5. The modes against real results (two panels)
6. Goalkeepers (three panels)

The other 42 figures are supplementary. Two of the six are new composites built for this
purpose; one of them, panel 2d, shows the distribution the whole paper is about, which no
figure previously did.

**"Some graphs still have the weird grey background."**
The 3D panes were the culprit, at 30 to 38 percent of those figures' area. All surfaces are
now pure white, verified by scanning every PNG.

**"You never want titles for graphs, the figure label underneath is the title."**
Every chart title is gone. What remains is a small unbolded panel header plus a bold panel
letter, and the caption spells out what each letter shows.

**"Graphs and figures are inconsistent in sizing and shape."**
Every figure is now drawn at exactly the 6.5 inch text width and placed at full width, so type
size is identical from figure to figure and nothing is scaled. That required re-laying out the
supplement so each float holds one tall figure or two short ones.

## Discussion and conclusion

**"I consider these the same section."** Merged.

**"You'll want a lot of citations here too."** It had none; it now has thirteen.

## Supporting information

**"Batch everything after the references into Supporting Information."**
Done, and taken further: it is a separate document with its own title page, numbered S1,
Figure S1, Table S1. The article and the supplement reference each other, so the article can
say "Section S3" and the supplement "Figure 2" and both resolve.

**"Figure 41, any reason we only look at Prem players?"**
Section S10 now opens by answering it. The Premier League figures are the coverage of the live
post-withdrawal pull that the robustness sample is built on, not a sampling choice. The paper's
claims rest on the primary sample, which is all five leagues and all five seasons, refitted
independently in each. Section S10.3 additionally repeats the reduced-sample pipeline across all
five leagues, so its generality is tested rather than assumed.

---

## On splitting into multiple papers

Keep this as one paper. Every section of the article is evidence for the single claim that two
modes exist and finer types do not: the null calibration establishes it, the replication rules
out a local artefact, the ablation rules out one feature block carrying it, the outcomes test
rules out "this describes the statistics, not the sport", and the goalkeepers are the same shape
in a disjoint feature space. Remove any one and a reviewer has a standing objection.

One piece is genuinely separable, and it is not the goalkeepers. The **possession elasticities**
are independent of the clustering entirely, they correct a convention in wide use since 2014
that nobody has measured, and they have their own data, bootstrap intervals, per-league refits,
functional-form test and the attacking against defending asymmetry. They currently sit in
Section S3 of a supplement, where nobody looking for them will find them. As a short methods
note they would be cited by people who would never read a clustering paper.

The goalkeepers are stronger as the second instance of the paper's pattern than as a paper of
their own: on their own they are one binary distinction plus one caution.

## Target venue

Aimed at **JSDSS**, the Journal of Statistics and Data Science in Sports, with **JQAS** as the
fallback. JSDSS fits because its stated principles are open access, sports-data scope and
reproducibility, it follows the JASA reproducibility guide, and it designates work meeting that
standard, which this repository is already built to. It sets no word limit. One thing to know:
JSDSS wants accepted manuscripts as a Quarto Manuscript with figure code inline, so that is an
acceptance-time conversion. JQAS's requirements are met or one switch away; it wants Harvard
author-date citations, and `paper/preamble.tex` carries both styles as a commented pair.

Journal of Sports Sciences is the wrong venue: roughly a 5,000 word ceiling, and a
sports-science readership for a paper whose contribution is that a standard validation practice
does not test what it is taken to test.

## What has not changed

Every reported number. `results/macros.tex` is identical to the version reviewed, so the
revision moved prose and figures and touched no result.

## What still needs a human

The prose is internally consistent but has not been read by either of us end to end since the
rewrite. The archetype names and the tactical readings are hypotheses about what the profiles
mean, and they are the part that most wants your eye.
