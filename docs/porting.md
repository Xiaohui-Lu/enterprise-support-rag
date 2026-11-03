# Porting to a new dataset

What carries over to a new knowledge base, what has to be redone, and why. Written for the question "what do you need to do to run this system on a new dataset?", in a project review or an interview.

## Do we need hand labels again?

Yes, but fewer. The judge isn't validated in general. It was validated on WixQA: Wix help-center questions, expert reference answers written as step-by-step procedures, and the answers our strategies wrote for them. Agreement is a property of the judge, the rubric and the data together, so it has to be measured again on new data. No one can tell whether it transfers without measuring it.

For example, on WixQA the hard cases were answers that took a different but valid route through the Wix menus, and the rubric has a rule for that (D-025). On a returns-policy dataset ("Can I return an opened item after 30 days?"), references are short conditions, and the hard cases become answers that get the condition partly right. A rubric written for procedures may grade those inconsistently, and only a new agreement check would show it.

What makes it cheaper the second time:

- **The panel goes first (D-028).** Four model configurations pre-label the sample, and a person labels only what they split on, plus a random audit. On WixQA that was 15 of 80 answers instead of 80.
- **A small check comes first.** Validate on 30–40 answers drawn by whole question. If kappa is clearly above 0.6 for every strategy, extend the check to the full sample. If not, revise the rubric and validate the revision on reserve questions (D-027).
- **The tools are ready.** `srag label-sample`, `panel`, `panel-labels` and `label --adjudicate` work unchanged once the loader exists.

## What the new dataset needs

| The dataset has | What follows |
|---|---|
| Questions, reference answers and gold articles (like WixQA) | The whole pipeline applies; validate the judge again |
| Questions and gold articles, no reference answers | Retrieval metrics work. The reference-guided judge has nothing to compare against, so write references for the dev set (or have a model draft them and a person check them), or grade groundedness only |
| Documents only ("bring your own docs") | No gold labels of any kind: demos, not results (experiment design, section 14). Build a small labeled dev set before claiming numbers |
| Unanswerable questions (like EnterpriseRAG-Bench) | A `NOT_FOUND` is no longer always wrong. D-025 grades a decline as incorrect only because every WixQA question is answerable, so the rubric needs a "correct decline" grade |

## Checklist, in order

1. **Loader:** map the data to `Article` and `Question` records (D-013). Check the license, and pin the dataset revision (D-011).
2. **Corpus statistics:** article lengths, the share of multi-article questions, and the kinds of questions. Long articles change chunking (D-012) and token cost; for example, Haiku charges more for prompts over 100K tokens.
3. **Split:** freeze a dev/test split by question before looking at any result (D-004).
4. **Retrieval baseline on dev:** check the tokenizer choices again. The stopwords and stemming are English, and stemming was decided on WixQA data (D-019).
5. **Prompts:** the answer and judge prompts name Wix. Rewrite them for the domain and bump `PROMPT_VERSION` and `JUDGE_PROMPT_VERSION`; old cache entries then simply stop matching.
6. **Rubric:** reread the hard-case rules for the new domain (alternative routes, declines, numbers), and bump the judge's version.
7. **Judge validation:** draw a new label sample by whole question, let the panel pre-label it, hand-label its splits plus an audit, and require kappa ≥ 0.6 per strategy (D-005, D-027, D-028). All of this happens before any test run.
8. **Tuning on dev:** set k, effort and router thresholds again. No number tuned on WixQA carries over.
9. **Cost:** run a pilot with `--limit`, estimate the full run, then run the test set once, at a milestone.

**What carries over:**
- **The code:** strategies, metrics, run records, the cache, the batch path, statistics and the labeling tools.
- **The method:** a pinned dataset, a frozen split, tuning on dev, and a judge validated before it's trusted.

**What doesn't:** any number. Thresholds, k, effort and kappa all have to be measured again.

## In short

"The code ports; the measurements don't. A judge's agreement with people is measured on one dataset, so on a new one I'd validate it again before trusting any score: a new label sample drawn by whole question, the model panel to pre-label it, hand labels only where the panel disagrees plus a random audit, and kappa per strategy. If the new data has no reference answers, the reference-guided judge doesn't apply, so the first job is writing references for a dev set. If it has unanswerable questions, the rubric needs a grade for a correct decline."
