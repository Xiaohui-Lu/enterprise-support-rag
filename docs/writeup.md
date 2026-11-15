# When do retrieval, agents and long context pay off?

*A write-up of a controlled study on a 6,221-article help center. Every number comes from the 350 held-out test questions unless it says dev. The [README](../README.md) has the tables and charts, and the [decision log](decisions.md) has every choice with its alternatives.*

## In short

- **Better retrieval was the biggest win, and it was cheaper.** Dense search instead of BM25 raised correct answers from 64% to 83% and cut the cost per answer by 28%.
- **An agent didn't pay here.** A search-and-read loop scored 81%, against 83% for one retrieval and one model call, and it was slower. These questions need one lookup.
- **Long context helped a little, at six times the cost.** Sending up to 100 articles scored 86.6%. The 3.7-point gain didn't survive the correction for comparing five strategies.
- **More text didn't hurt.** Up to 90,000 tokens, accuracy didn't fall. Look-alike articles helped rather than distracted.
- **A router on cheap signals barely beat chance.** It scored 84.9%, only 0.7 points above sending the same share of questions to long context at random. Choosing perfectly, with hindsight, would score 90%.

## The question

A support team can answer a question with an LLM in four ways:
- from the model's memory alone;
- from a few retrieved articles;
- with an agent that searches and reads;
- by putting a large slice of the help center into one long prompt.

Their costs and speeds differ by more than an order of magnitude. This study measured which one pays off. It used WixQA: 6,221 Wix help-center articles and 400 real questions, each with an expert-written reference answer.

The setup was built to keep the comparisons fair:
- **One generator for every strategy:** Claude Haiku 5.5, with the same prompt and the same answer format. So any difference comes from what the model is shown.
- **A judge validated before use.** Claude Sonnet 5.5 grades each answer against the reference. Before it was trusted, it agreed with 80 labels at Cohen's kappa 0.82.
- **Tuned on 50 dev questions, judged on 350 held-out ones.** The test set ran once per milestone, from a clean commit. The rule for each verdict was written down before the run.
- **Paired statistics.** Every comparison is on the same questions, with a bootstrap interval and McNemar's test. Holm's correction applies when several strategies compete.

## What was found

### 1. Retrieval quality mattered most

BM25 keyword search put a gold article in its top 10 for only 57% of test questions. The reason is wording: real customers don't use the articles' words.
- On synthetic questions written from the articles, BM25's recall@5 on dev was 0.90.
- On real questions, it was about 0.53.

Dense search with a small open-weight embedding model, voyage-4-nano, fixed most of that.
- Accuracy rose from 64.0% to 82.9%: +18.9 points [+14.0, +23.7]. It fixed 80 questions and broke 14.
- It was also cheaper, because its top 10 articles are shorter.

The usual upgrades added nothing on top, in tests on dev:
- Hybrid fusion with BM25 lost 3 to 7 points of recall.
- Two rerankers didn't beat dense search's own order.
- Searching by section looked 5 points better, but that rested on 3 questions, short of the bar.

A support team's first investment should be the retriever.

### 2. The "right" articles aren't the ceiling

Giving the model exactly the gold articles, an oracle, scored 74.6%. Dense search's ten articles scored 82.9%, 8.3 points more. Neighbouring articles often carry the part of the answer the gold one leaves out. So "gold article in the top k" understates what retrieval delivers.

### 3. The agent didn't pay

S2 is an agent loop written from scratch, with no framework:
- two tools, `search` (numbered snippets) and `read` (one full article);
- limits on steps and prompt tokens;
- prompt caching.

How it compared with single-pass retrieval:
- **Accuracy:** 81.1%, 1.7 points lower [−5.7, +2.0].
- **Latency:** a median of 5.1 s, against 3.7 s.
- **Cost:** 17% cheaper, but only because prompt caching discounts the history it resends on each call.

It took two tool steps on 291 of the 350 questions: one search, then one read. When the first search already finds the answer, an agent adds a turn, not insight. An agent should pay on questions that need several hops, and WixQA rarely asks those.

### 4. Long context: a small gain at a large price, and no rot

S3 sent the top 20, 50 or 100 articles in one prompt, capped below 90,000 tokens to stay under the price line. It scored 82.9%, 85.1% and 86.6%.
- The best is +3.7 points [+0.6, +7.1] over single-pass retrieval.
- With five strategies competing, though, Holm's correction puts its p at 0.19, so it didn't replace the default.
- If the gain is real, it costs about $0.18 per extra correct answer. That's cheap next to a ticket a person handles, so it's worth a confirmatory test.

The context-rot test buried each question's gold articles among 10,000 to 90,000 tokens of other articles. Accuracy didn't fall:
- With random articles around the gold, it held at 76% to 77%.
- With look-alike articles, it scored 7 to 11 points higher than with random ones, at every size, because most look-alikes carry part of the answer.
- There was no "lost in the middle". With 90,000 tokens of look-alikes, gold articles in the middle third of the prompt did best: 91.5%, against 80.2% in the first third and 84.0% in the last.

This differs from [Chroma's 2025 study](https://research.trychroma.com/context-rot), where similar distractors hurt. Theirs were built to mislead. These are real articles from the same help center.

### 5. A router needs better signals than retrieval scores

The router, R2, answered with single-pass retrieval. It escalated to long context when the top retrieval score was weak, or when the first answer declined. It scored 84.9%.
- R2 sent 34% of questions to long context. Sending the same share at random is expected to score 84.1%. R2 beat that by only +0.7 points [−0.9, +2.5].
- The weak-score rule alone added 0.2 points over chance.
- A decline carries real information, but it happened on only 11 of 350 questions.

The room exists, though. Choosing between the two strategies with hindsight would score 90.0%, at about a fifth of long context's cost. Finding those questions needs a better signal than a similarity score, such as a model judging whether its context is enough, at the price of another call.

## What the method taught

- **50 dev questions can't see a 10-point gap.** At the first milestone, dev showed retrieval level with the oracle: +2 [−14, +18]. The 350 test questions showed a 10.6-point gap. Later, in the router step, only one dev question could be fixed by routing at all. A small dev set catches bugs and big effects. It can't tune.
- **Dev flatters.** Settings chosen on dev scored 7 to 12 points lower on test: BM25 went from 76% to 64%, and dense search from 90% to 82.9%.
- **Write the bar down before the run.** Every verdict above came from a rule committed before the test run. Without one, long context's raw p of 0.047 would have been easy to call a win.
- **Compare against chance, not just the baseline.** The router beat the default by 2 points. Only the random-escalation control showed that most of that came from sending more questions to a stronger strategy, not from choosing them well.
- **A cache makes evaluation exact.** Identical requests return identical answers. So re-running a report costs nothing, and the CI gate can tell for certain whether a change altered anything the model sees.
- **Check the stopwatch.** The first answer of every run took 27 to 30 s. It wasn't the API: a cold import of the SDK, inside the timed region, took 23.5 s.

## From experiment to service

The evaluation calls the same functions the service runs, so the numbers describe what ships.
- **The service:** FastAPI streams the answer as Server-Sent Events to a small React page, which links each citation to its article.
  - In the runs measured, the first words arrived after 0.6 to 2.2 s.
  - A repeated question comes back from the cache in 0.14 s.
- **Deployment:** Docker Compose runs it, with the model and the articles baked into the image at pinned revisions.
- **CI:** lint, 420 Python tests and the page's tests and build. CI also refuses code whose eval gate hasn't passed.
  - First, the gate checks whether the request for any of 50 dev questions changed. That check is free and has no noise.
  - Only the questions whose requests changed are re-answered and graded.

## Limits

- **One generator, one judge, one corpus.**
- **The judge leans lenient,** by about 4 points in absolute terms. Differences between strategies are affected less.
- **Some validation labels weren't audited.** 65 of the judge's 80 validation labels came from a unanimous panel of Claude models, which can share the judge's blind spots, and weren't checked by hand.
- **Each strategy ran once.** The intervals cover which questions were drawn, not the model's sampling.
- **"Correct" means agreeing with WixQA's reference answer,** and a reference can itself be incomplete.
- **Results by question type are directional.** They rest on 70 to 175 questions each.

## Cost

The study cost about $51 in API calls:
- $11 for the labeling panel;
- $5 for the first milestone;
- $29 for the second;
- about $6 for dev work.

Everything after the second milestone cost about a cent, including the router's test run, because it replayed cached calls.

## What next

- **A confirmatory run of long context at K=100** on fresh questions, to settle whether its 3.7 points are real.
- **A benchmark with multi-hop questions,** where an agent should beat single-pass retrieval: the mirror image of finding 3.
- **A support agent that acts as well as answers,** measured on a benchmark built for agents.
