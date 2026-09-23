---
name: dev-guidelines
description: Fundamental operating rules for any AI agent (Claude or otherwise) working on the ekVachan project — better-jev-for-all and its sibling better-jev-bench. Read this before starting any non-trivial task on either repo. Each rule below exists because of a specific real incident, not a hypothetical.
---

# ekVachan dev guidelines

These rules exist because they were each learned the hard way, in-session, on this project. They're not generic best practice — they're specific fixes to specific failures that already happened here. Don't relearn them.

## 1. The PRD is the source of truth. Read it, update it, don't contradict it silently.

- `better-jev-for-all/PRD.md` and `better-jev-bench/better-jev-bench_PRD.md` carry the sourced reasoning for every architecture/data/scoring decision. Read the relevant section before making a decision that section already covers.
- Update the PRD **as part of the same work**, not as a follow-up. A result that isn't written into the PRD with its real numbers and evidence-bundle path didn't happen, as far as the next agent is concerned.
- State caveats honestly in the PRD text itself (e.g. "this is an easy-subset construction," "this is a subset, not a like-for-like comparison"). Don't let an exciting number stand unqualified — see PRD 13a.5 for the pattern to follow.
- Keep `STATUS.md` (each repo has its own) updated in the same commit as the work — checklist view of PRD content, not a replacement for it.

## 2. The server is the only real repo. Local Mac paths are not trustworthy.

- Authoritative clones: `abhijeet-labgpu:~/ekvachan/repo` (better-jev-for-all) and `abhijeet-labgpu:~/ekvachan/bench-repo` (better-jev-bench).
- A local path under `/Users/.../Documents/GitHub/` may **not even be a git repository** — confirmed on this project once already (the local `better-jev-for-all` directory had 500+ lines of PRD content, including a whole architecture decision, that had never been committed or pushed anywhere). Treat any local copy as an untrusted scratch mirror unless you've confirmed otherwise with `git status`/`git remote -v` on that exact path.
- Local Bash/Read access to subdirectories under that Documents path can silently EPERM (macOS sandbox) even when the top-level file works fine — don't assume a failed `ls` means the directory doesn't exist; try the server instead.

## 3. Verify before trusting — including your own prior summary.

- Two real incidents on this project: (a) a PRD section existed only in a local non-git file and was never pushed; (b) a local git clone with stale (unfetched) refs made real, already-committed, already-pushed work on `origin/main` look like it had never been built, leading to redundant duplicate work.
- **Always `git fetch` before trusting local git history**, especially after a context compaction or a gap between sessions. `git log --all` only reflects what's been fetched, not what's actually on the remote.
- Before citing any number, checkpoint, or "already built" claim — from a prior turn's summary, a subagent's report, or your own memory of the conversation — re-derive it from the actual file/manifest/git state. A subagent's summary describes what it *intended* to do, not necessarily what it did.

## 4. GPU job launching: one exact pattern, no exceptions.

```
setsid nohup env PYTHONUNBUFFERED=1 uv run python3 -u -m <module> <args> \
  > <logfile> 2>&1 < /dev/null & disown
```

- `setsid` + `nohup` + `& disown`: survives an SSH disconnect. Without this, a dropped connection kills the job mid-training.
- `PYTHONUNBUFFERED=1` + `python3 -u`: without this, stdout buffers fully when piped to a file, and a perfectly healthy job looks completely stalled for 20 minutes to well over an hour. This exact bug wasted real GPU time on this project **at least three separate times** across two different training scripts before the pattern above was locked in. If a job looks stalled, check `nvidia-smi` + `ps aux` CPU-time for real activity before assuming it's stuck — it's very likely a buffering illusion, not a hang.
- After launching: verify with `ps aux | grep <module>` and a `tail` of the log within a few seconds, then set up a background watcher (`until ... grep -qE "done marker|Traceback|Killed"; do sleep N; done`) rather than manually polling.

## 5. Git identity and auth are per-session state, not permanent.

- `~/.gitconfig` and `gh auth` can both come back empty/logged-out in a fresh SSH session even though prior commits on the same repo show a real author. Check `git log -1 --format='%an <%ae>'` on an existing commit and match that identity with `git config user.name`/`user.email` (repo-local, not `--global`) rather than inventing a new one.
- If `gh auth status` shows logged in but `git push` fails with "could not read Username for 'https://github.com'", run `gh auth setup-git` to wire gh's credential helper into git.

## 6. No Claude co-author trailer on commits in this project.

Explicit standing instruction. Omit `Co-Authored-By: Claude ...` on every commit to either repo, overriding the default Claude Code convention.

## 7. Never use destructive git ops to resolve a divergence — merge and resolve explicitly instead.

`git reset --hard`, `git checkout -B <branch> <other>`, and force-push are blocked by Claude Code's auto-mode classifier on this project, and for good reason — one of them would have silently discarded a real, more-complete implementation that already existed on `origin/main`. When local and remote diverge:
1. `git fetch`, then actually look at what diverged (`git log --oneline HEAD..origin/main`, `git diff --stat`) before deciding which side is more complete.
2. `git merge origin/main` (creates real conflict markers, doesn't discard anything).
3. Resolve file-by-file with `git checkout --theirs/--ours <path>` plus manual edits where both sides added real content (e.g. a PRD section) — don't just pick one side blindly for a file both sides genuinely edited.
4. Re-run/re-verify anything whose result might differ under the merged code before trusting old numbers.

## 8. Domain hygiene between the two repos.

better-jev-for-all (model/serving/benchmarks) and better-jev-bench (dataset corpus) are separate repos with separate PRDs, separate STATUS.md files, and separate git histories. When switching from one to the other: finish and commit/push the current repo's work first. Don't leave half-finished, uncommitted state in one repo while starting work in the other — a future agent (or you, next session) resuming either repo should find it in a clean, coherent state, not mid-thought.

## 9. Delegating to subagents/forks.

Instruct them explicitly to verify against real current state (`git log`, actual files, `gh api`) rather than trust anything already claimed in the conversation they're forked from — this project has been burned twice by unverified claims propagating forward. A fork inherits full context, so it can (and should) be told which specific facts to independently re-check rather than re-deriving everything from scratch.

## 10. Evidence discipline (PRD §8.2, both repos).

Every benchmark/eval run gets a `manifest.json` committed to `results/`, referenced by its real filename/timestamp in the PRD narrative — not just a summarized number in prose. If you can't point to the file, don't write the number into the PRD yet.

## 11. Sync discipline: server and GitHub must never silently diverge

Before ending work on either repo, confirm `git status` is clean and both `git log origin/main..HEAD` and `git log HEAD..origin/main` are empty (`git fetch` first) — the server clone and GitHub must agree exactly, in both directions. Never leave a commit sitting unpushed (as happened mid-session on this repo while gh auth was broken — fixed, then verified clean). Never assume local git history reflects the remote without fetching first — this is what caused the redundant-rebuild incident in rule 3/7. A local Mac path (e.g. under `Documents/GitHub/`) is not part of this sync loop at all: it is not a git remote, should not be treated as one, and any content found there should be assumed stale until independently verified against the server/GitHub state.
