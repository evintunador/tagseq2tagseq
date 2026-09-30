# tagseq2tagseq closeout plan

Horizon: cluster compute available until mid-to-late October.

Everything is on `main`. There are no open pull requests and no unmerged work in any
worktree. What remains is a set of decisions and a compute budget.

---

## 1. State

| thread | state | what is left |
|---|---|---|
| merged_v2 ladder | complete, on main | optional hardening, see 4b |
| eval-run-tracking | mechanism on main | the RepoBench decision, then quarantine |
| link-injection | scaffold and sciq results on main | a real retriever rung |
| epochs-to-degradation | drifted, partly invalidated | scope decision, see 4d |
| sparsity | complete, on main | nothing |
| provenance and grounding | complete, on main | clear the remaining debt |
| paper | 49-page draft on main | items 2 and 3 |

Worktrees still on disk: `-memexp` (in use, see below), `-evaltrack`, `-linkeval`,
`-sparsity`. All four are fully merged into main. Only `-memexp` is live: the yield
watcher and the running jobs launch from it, so do not rebase or switch branches there.
The `-sparsity` worktree has a python process attached to it despite its branch being
merged since August.

Branches not reachable from main: `run-provenance-artifacts` and seven `worktree-agent-*`
refs. Their content is present in main, but the commits themselves are not ancestors, so
they need a force delete rather than a safe one. Left alone deliberately.

### What is running

Four single-node jobs, all the same arm: java, `doc_causal`, 16 epochs, 33792 steps, at
muon learning rates spanning 1e-3 down to 2.5e-5. Combined with the arms that finished
earlier at 4e-3, 3e-3 and 2e-3, this sweep now covers more than two orders of magnitude on
one configuration. The earlier three were monotone: 2e-3 gave the best validation loss,
recovering the 8-epoch floor, while 3e-3 reproduced the degraded 16-epoch number.

This sweep is the entire current compute spend. Section 4 argues most of it should move.

---

## 2. The paper cites numbers that are known to be wrong

`paper/generated/values.tex` defines `compute.repobench_ppl.*` as 7.25 / 8.94 / 8.76 /
10.4, and `paper/sections/06_results.tex` renders them seven times, including the
compute-control table. Re-running under isolated eval run dirs collapses all four to
roughly 5.9 with overlapping intervals.

The cause is structural, not a bug in the benchmark. `run_repobench` scores every model
under `doc_causal`, which is correct for a flat single-doc baseline but means the
cross-doc mechanism is off at scoring time. Four training masks therefore cannot be
distinguished by it. `check_grounding.py` passes anyway, because the ledger's `expected:`
still holds the old values; it cannot detect this class of drift.

Three options:
- **(a) Drop** the four-way compute-control claim.
- **(b) Re-ground** on the within-`cross_doc_link` `repobench_cross_doc` delta, which
  reproduced exactly: Java 1.383 cross against 1.448 flat, Python 1.700 against 1.792.
- **(c) Build** the mask-aware version by relaxing `run_repobench_cross_doc` so the packed
  layout is scored with each model's own mask. Filed under Eval in `TODOS.md`. Needs a
  decision about what a `doc_causal` model uses for `link_detector`.

Recommendation: **(b), and drop the compute-control framing**. It costs nothing, it is
honest, and it agrees with the sparsity result that the link benefit is overwhelmingly an
inference-time effect. Treat (c) as optional.

Once the decision is applied, run the quarantine script in `scripts/rerun/` to retire the
old contaminated eval sidecars. Its distiller is on main, so the ordering constraint that
used to apply is satisfied.

---

## 3. Paper gaps that cost nothing

- 30 `\fillin{}` blanks: 7 in datasets, 10 in results, 1 in the abstract
  ("strengthens/weakens?"). The diversity-scaling blanks are answerable from the finished
  ladder.
- 11 `literal` ungrounded-debt ledger entries: two traversal validation losses, eight
  step-time, speedup and coverage numbers taken from prose whose raw CSVs were never
  located, and one cross-run regression fit.
- 16 `singledoc.*.ci` keys defined but never cited.
- No LaTeX toolchain on this host, so buildability is unverified. Worth one check
  elsewhere.

---

## 4. What to spend the remaining compute on

Ranked by value per node-hour.

**4a. Memorization probe on the two 32B-balanced checkpoints.** Eval only. Both
32B-balanced arms degraded mid-run at peak learning rate and only the cross-doc arm
recovered, so that pair is excluded from the "cross-doc training is free" claim. The probe
separates the two candidate explanations, too-hot learning rate against a data-repeat
memorization effect, and it is the one measurement that serves both the ladder and the
epochs-to-degradation experiment. Cheapest load-bearing item here.

**4b. Two extra seeds of the 3.9B cross_doc arm.** Every rung is a single seed and the
headline claim is that the cross-doc delta is *flat* across eight times the tokens. Flat
against an unmeasured noise floor is the weakest point in the paper. The results doc
already concedes a 0.03 to 0.05 wobble and then dismisses a +0.43 outlier as noise after
the fact. Two seeds at the cheapest rung fix this.

**4c. Link-injection: one real retriever rung.** The current "retrieved" condition is junk,
with 302 of 999 titles literally "?", so the negative interaction measures junk tolerance
rather than retrieval. A BM25 or entity-match rung is about one GPU for a couple of hours
and turns that thread into a result. The stronger matched pair it also wants is a new
training run and needs explicit approval.

**4d. Epochs-to-degradation: finish or freeze.** The learning-rate sweep now running is
the whole question. If the 16-epoch degradation disappears at a low enough rate, the
experiment's headline was an optimizer artifact and the 12- and 16-epoch points on both
masks need re-running at the chosen rate before they mean anything. That is four
multi-day runs. Recommendation: **stop the sweep once it brackets the floor**, take the
best rate, and decide then. Do not keep four nodes on one arm while 4a and 4b are unfunded.

Optional if nodes are free: re-port the specialists with flat nll, and a wiki
community-pack grant check.

---

## 5. Where to be skeptical

1. **The paper's RepoBench numbers are wrong and still published.** Section 2.
2. **Single seed everywhere in the ladder.** "Flat across scale and diversity" is asserted
   against a noise floor that was never measured.
3. **"Matches or beats specialists on 9 of 13 ports" counts three ties as wins** (6 win, 3
   tie, 4 loss), and the specialist numbers come from a different lineage with different
   harness discipline. The headroom confound is acknowledged and unresolved.
4. **The retracted headline still sits at the top of the merged_v2 results doc** behind a
   banner rather than being removed. That family was retracted for a
   sequential-not-interleaved dataloader bug. A reader going top-down meets dead numbers
   first.
5. **Epochs-to-degradation drifted into a different experiment than the one designed.** The
   corpus changed from go to java; the cross-doc hard-repetition baseline was dropped, so
   fresh-against-repeat is no longer tested; the headline metric moved from train/validation
   gap to absolute validation loss; the eval checkpoint moved from latest to best; and the
   verbatim-recall probe, the concat masks and the standard-pack baseline were never run.
   Each change was defensible alone. Together they mean the current figure does not answer
   the original question.
6. **That experiment's headline degradation is probably an optimizer artifact.** Hence the
   sweep.
7. **Its mask separation sits inside the noise.** Intervals are about plus or minus 0.010
   and the 8-epoch gap is 0.0146.
8. **Roughly 60 watcher yield-and-resume cycles were never re-audited.** Resume correctness
   was verified once in August, before the churn. An earlier watcher bug caused 120 silent
   full resets against 163 real resumes, so pre-September trajectories contain
   restarted-from-zero segments.
9. **A latent watcher bug can still lose runs.** A yielded job whose cancel return code is
   swallowed never reaches the yield ledger and so never auto-resumes. A related failure
   appears in the log as a job that could not be mapped back to its run directory.
10. **The link-injection gold interaction is fragile.** +0.27 nats against a +5 main effect,
    heavy-tailed with a +0.10 median, on an undertrained pair, after mid-flight swaps of
    both the benchmark and the grant-detection method.
11. **Most configs in the repo still set `train_loop.warmup_steps`,** which is no longer
    read. Those runs now fail loudly with the equivalent percentage in the error rather than
    silently training with no warmup, but they do need the one-line swap before reuse.

---

## 6. Backlog hygiene

`TODOS.md` now carries the items that previously lived only in the status docs. It also
still carries work that is finished. These sections are verified complete or superseded
and can be struck; the evidence is a merged PR, a results table or the code itself:

wiki community_pack audit · merge-all-datasets diversity scaling · edge-dropout density
line · TheStack other languages · Stack Python link resolution · merged composite link
resolution · retune LR and schedule · train the ablation matrix · integrate easy LLM
benchmarks · synthetic intra-repo benchmark · create Go/Java benchmarks · self-built
benchmark from test_community · better multi-language benchmark · link injection eval for
external benchmarks · resume latent bugs.

Superseded rather than done: the diversity section's blocker is cleared, the batch-size
sweep was self-deferring, and two bullets inside the cross-doc port follow-ups are dead
(the Kotlin sample-size raise is impossible against a capped public pool, and the Go
adapter was removed).

One is genuinely unclear: the short LR check at the 16B rung. Sweep configs exist and an
audit ran, but the results doc still says LR and weight decay were never retuned across
rungs, and the cross-rung claims rest on that. Worth resolving before the paper cites
cross-rung deltas.

Recovered reasoning from the unopenable sessions is in `docs/EVAL_DECISIONS.md`.

## 7. Housekeeping

### Where the data lives

All data and run outputs live under `/fss-data/evin_t/tagseq2tagseq_artifacts/`. The main
checkout holds only code; it keeps symlinks at the old paths so that provenance records and
result files embedding those paths still resolve.

| checkout path | fss-data location |
|---|---|
| `runs` | `runs_legacy` (runs written before the fss-data runs root existed) |
| `schedules` | `schedules_legacy` |
| `evals`, `artifacts` | `evals`, `artifacts` |
| `logs`, `slurm_logs` | `pipeline_logs/repo_logs`, `pipeline_logs/repo_slurm_logs` |
| `data/.cache` | `cache/main_checkout_data_cache` |
| `data/wiki_articles*` | `raw/wiki_articles*` |
| `data/github_graph_extractor/{graph,sample}*` | `graphs/github_graph_extractor/` |

`runs_legacy` is kept separate from `runs` on purpose. The quarantine script selects
targets by a timing heuristic across every run under its roots, and the legacy set is the
one known to hold contaminated eval sidecars. All 22 run ids cited by
`provenance/ledger.yaml` are in `runs_legacy`, including the matched pair the
link-injection evaluation scores against.

The virtual environment stays in the checkout, since a venv cannot be relocated.

**Launch hazard:** the reproducibility manager reads every untracked, non-ignored file into
its patch when a run starts. A large untracked file anywhere in a checkout silently bloats
or hangs the next run launched from it. Keep bulk data out of checkouts or ignored.

### Reclaimable space in runs_legacy

Every run stores both `best_model.pt` and `latest.pt`, and 373 runs have both.
Best-checkpoint selection ran on a validation metric that was itself buggy, which is why
evaluation standardised on the fully cooled `latest.pt`, so `best_model.pt` is dead weight
wherever `latest.pt` exists. Dropping it everywhere except the 22 ledger runs reclaims
**1.31 TB**. 48 runs have only `best_model.pt`; do not touch those, or they lose their only
checkpoint. The distilled records in `provenance/runs/` keep run metadata independently of
the checkpoints.

### Other

- `graphs/github_graph_extractor/sample_{1M,10M,100M}.jsonl` is 131 GB, and those graph keys
  predate the normalization refactor, so they may be stale as well as large.
- Recreatable caches of a few GB each under `-evaltrack` and `-memexp`.
- `aws_keys_scratch.txt` sits in the home directory. Worth rotating or deleting.

### Branches

Not cleaned up. Four merged branches are held open only by their worktrees
(`eval-run-tracking`, `link-injection-eval`, `memexp-isolated`, `sparsity-scaling-law`);
removing a worktree frees its branch for a safe delete, but `-memexp` is live and must stay.
Eight branches are not ancestors of main (`run-provenance-artifacts` and seven
`worktree-agent-*`), each carrying one commit whose content reached main by another route,
so they need a force delete rather than a safe one.

---

## 8. Order

1. Make the RepoBench call and apply it. Run the quarantine.
2. Launch 4a and 4b. Wind the learning-rate sweep down to what still answers 4d.
3. Fill the diversity blanks from the finished ladder.
4. Fund 4c if nodes allow.
5. Paper only: verify the build, clear the grounding debt, remove or fold the retracted
   section, final artifact backup.
