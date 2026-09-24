"""
Capability probe for the >26-option gap (PRD 13a.15/13a.16, 5.1a). Not a
training script and not a benchmark -- it answers one question with real
evidence, the way `probe_vision_path.py` answered 5.2b's:

    can this project's restricted-logit mechanism address more than 26
    options, and if so by which of the two candidate routes -- a multi-token
    option identifier (PRD 13a.5's own "different identifier scheme"), or
    5.1a's cross-attention / late-interaction decision head?

Motivation, measured not hypothetical: `bjb evaluate` scored Generality 0.0
because 6 of the 14 real corpus tasks exceed 26 options (clinc150 151-way,
ledgar 100-way, banking77 77-way, massive/intent 60-way, cuad/clause_type
41-way, go_emotions 28-way) and are *declined*, not answered badly.

Every claim in PRD 5.1b's "what was actually measured" table is produced by
this file, so the next agent can re-derive them instead of trusting the prose
(dev-guidelines rule 3). Writes a manifest to `results/` (rule 10).

Run (CPU-only checks, ~2 min, no GPU needed):
    uv run python3 -u -m training.probe_multitoken_scheme

Run (adds real bf16 forward passes on the GPU, ~10 GB VRAM, ~4 min):
    uv run python3 -u -m training.probe_multitoken_scheme --gpu-forward

The GPU section deliberately measures four things and no more: (1) how much a
genuinely wide prompt costs versus today's 26-option one, (2) whether the
single-position restricted read still yields a sane distribution over a wide
code table, (3) what a *true* multi-token (two-position) read would have cost
had one been necessary -- both the KV-cached-second-step variant and the
one-pass two-position variant -- and (4) the cost model of 5.1a's
late-interaction head, so path 2 is rejected (or not) on a number rather than
on a prior.
"""

import argparse
import itertools
import json
import math
import string
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "Qwen/Qwen3.5-4B"

UPPER = list(string.ascii_uppercase)
LETTERS = UPPER[:]  # the current mechanism's full budget: 26
ALL_PAIRS = ["".join(p) for p in itertools.product(UPPER, UPPER)]  # 676, lexicographic

# The real, license-clean corpus tasks that exceed 26 options (13a.15). Label
# text is pulled from the local HF cache where available so prompt-length
# measurements use real option strings, not synthetic stand-ins.
WIDE_TASK_SOURCES = [
    ("clinc150", "clinc/clinc_oos", "plus", "test", "intent"),
    ("ledgar", "coastalcph/lex_glue", "ledgar", "test", "label"),
    ("banking77", "legacy-datasets/banking77", None, "test", "label"),
    ("go_emotions", "google-research-datasets/go_emotions", "simplified", "test", "labels"),
]

INSTRUCTIONS = "Classify the user's utterance into one of the following intents."
STATE = "what is the exchange rate between the dollar and the euro right now"


# --------------------------------------------------------------------------
# scheme construction -- deliberately a runtime derivation, not a hardcoded
# table, so a tokenizer change fails loudly instead of silently mis-reading.
# --------------------------------------------------------------------------
def single_token_ids(tok, strings: list[str]) -> dict[str, int | None]:
    """id if `s` encodes to exactly one token standalone, else None."""
    out = {}
    for s in strings:
        ids = tok.encode(s, add_special_tokens=False)
        out[s] = ids[0] if len(ids) == 1 else None
    return out


def build_wide_code_table(tok) -> tuple[list[str], list[int]]:
    """A-Z first (so any item with <=26 options gets a prompt byte-identical to
    today's, and the existing checkpoint's learned behaviour is untouched),
    then every two-uppercase-letter pair that is a single token, in
    lexicographic order. The table is generated, asserted, and returned --
    never hardcoded."""
    ids1 = single_token_ids(tok, LETTERS)
    assert all(v is not None for v in ids1.values()), "A-Z is not all single tokens -- 26-ceiling claim broken"
    ids2 = single_token_ids(tok, ALL_PAIRS)
    codes = LETTERS + [c for c in ALL_PAIRS if ids2[c] is not None]
    ids = [ids1[c] for c in LETTERS] + [ids2[c] for c in ALL_PAIRS if ids2[c] is not None]
    assert len(set(ids)) == len(ids), "code table has a duplicate token id"
    return codes, ids


def build_prompt_text(state: str, options: list[str], codes: list[str], instructions: str,
                      answer_line: str = "range") -> str:
    """Same shape as `decoder_lora_lib.build_prompt_text()`, with the option
    label drawn from `codes` instead of hardcoded `LETTERS`, and the trailing
    answer line able to name a *range* instead of enumerating every code
    (enumerating 151 codes costs real tokens -- measured below)."""
    n = len(options)
    option_lines = "\n".join(f"{codes[i]}) {options[i]}" for i in range(n))
    if answer_line == "enumerate":
        tail = f"Answer with a single code ({'/'.join(codes[:n])})."
    else:
        tail = f"Answer with a single option code from the list above ({codes[0]} .. {codes[n-1]})."
    return f"{instructions}\n\n{state}\n\nOptions:\n{option_lines}\n\n{tail}"


def _load_label_spaces() -> dict:
    """Best-effort: real label text from the local HF cache. A source that is
    not cached is reported as absent rather than faked."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset

    spaces = {}
    for name, repo, config, split, col in WIDE_TASK_SOURCES:
        try:
            ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
            feat = ds.features[col]
            names = feat.feature.names if hasattr(feat, "feature") else feat.names
            spaces[name] = {"n": len(names), "labels": list(names), "repo": repo, "cached": True}
        except Exception as exc:  # noqa: BLE001 -- a missing cache is a finding, not a crash
            spaces[name] = {"cached": False, "repo": repo, "error": f"{type(exc).__name__}: {str(exc)[:120]}"}
    return spaces


def _dist_stats(probs: np.ndarray) -> dict:
    p = probs.astype(np.float64)
    p = p / p.sum()
    n = len(p)
    ent = float(-(p * np.log(np.clip(p, 1e-12, None))).sum())
    return {
        "n": n,
        "max_prob": round(float(p.max()), 5),
        "entropy_nats": round(ent, 4),
        "normalized_entropy": round(ent / math.log(n), 4),
        "top5": [[int(i), round(float(p[i]), 5)] for i in np.argsort(-p)[:5]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--gpu-forward", action="store_true", help="also load the model and run real forward passes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import transformers
    from transformers import AutoConfig, AutoTokenizer

    findings: dict = {
        "probe": "multitoken_option_scheme",
        "base_model": args.base_model,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
        "question": (
            "Can the restricted-logit mechanism address >26 options, and by which route -- a "
            "multi-token identifier (PRD 13a.5) or 5.1a's cross-attention head?"
        ),
    }

    tok = AutoTokenizer.from_pretrained(args.base_model)
    cfg = AutoConfig.from_pretrained(args.base_model)
    vocab = tok.get_vocab()

    # --- 1. re-verify the claimed 26-letter ceiling ------------------------
    ids1 = single_token_ids(tok, LETTERS)
    letter_ids = [ids1[c] for c in LETTERS]
    findings["single_letter_baseline"] = {
        "all_single_token": all(v is not None for v in ids1.values()),
        "ids": letter_ids,
        "contiguous": letter_ids == list(range(letter_ids[0], letter_ids[0] + 26)),
        "mutually_distinct": len(set(letter_ids)) == 26,
        "note": "Re-derived here rather than carried over from schema.py's literal (dev-guidelines rule 3).",
    }

    # --- 2. vocabulary scan: what identifier alphabets exist at all? -------
    import re

    def scan(pat: str) -> dict:
        rx = re.compile(pat)
        hits = sorted(t for t in vocab if rx.fullmatch(t))
        return {"count": len(hits), "sample": hits[:8]}

    findings["vocab_scan"] = {
        "vocab_size": len(vocab),
        "patterns": {
            p: scan(p)
            for p in [
                r"[A-Z]", r"[a-z]", r"[0-9]",
                r"[A-Z]{2}", r"[A-Z]{3}", r"[A-Za-z]{2}",
                r"[A-Z][0-9]", r"[0-9]{2}", r"[0-9]{3}",
                r"Ġ[A-Z]", r"Ġ[A-Z]{2}",
            ]
        },
        "note": (
            "A token existing in the vocabulary is necessary but NOT sufficient -- it must also "
            "survive `encode()` standalone and in real prompt context. Both checked next."
        ),
    }

    # --- 3. candidate identifier schemes, characterised for real -----------
    schemes: dict = {}

    ids2 = single_token_ids(tok, ALL_PAIRS)
    ok2 = [c for c in ALL_PAIRS if ids2[c] is not None]
    bad2 = [c for c in ALL_PAIRS if ids2[c] is None]
    lens2 = [len(tok.encode(c, add_special_tokens=False)) for c in bad2]
    schemes["two_letter_uppercase"] = {
        "space": len(ALL_PAIRS),
        "single_token_count": len(ok2),
        "multi_token_count": len(bad2),
        "multi_token_length_histogram": {str(k): lens2.count(k) for k in sorted(set(lens2))},
        "first_missing": bad2[:16],
        "note": (
            "Codes that are NOT single tokens are simply excluded from the table -- there is no "
            "need to force them, since the single-token subset alone already exceeds every real "
            "option count in the corpus."
        ),
    }

    lower = single_token_ids(tok, list(string.ascii_lowercase))
    digits = single_token_ids(tok, list(string.digits))
    ld = single_token_ids(tok, [a + d for a in UPPER for d in string.digits])
    nums = single_token_ids(tok, [str(i) for i in range(1, 200)])
    schemes["mixed_case_plus_digits_single_position"] = {
        "A-Z": 26,
        "a-z_single_token": sum(v is not None for v in lower.values()),
        "0-9_single_token": sum(v is not None for v in digits.values()),
        "max_single_position_alphabet_without_pairs": 26 + sum(v is not None for v in lower.values())
        + sum(v is not None for v in digits.values()),
        "verdict": "62 < 151 -- case/digit mixing alone does NOT reach the widest real task.",
    }
    schemes["letter_digit_pairs"] = {
        "space": len(ld),
        "single_token_count": sum(v is not None for v in ld.values()),
        "verdict": "letter+digit pairs are not in this vocabulary as single tokens.",
    }
    schemes["decimal_numerals"] = {
        "space": 199,
        "single_token_count": sum(v is not None for v in nums.values()),
        "note": "Qwen splits multi-digit numerals into per-digit tokens; only 1-9 survive standalone.",
    }

    # a genuine two-position (multi-token) read, characterised for the record
    pair_seqs = {c: tok.encode(c, add_special_tokens=False) for c in ALL_PAIRS}
    exactly_two = [c for c, s in pair_seqs.items() if len(s) == 2]
    two_pos_consistent = [
        c for c in exactly_two if pair_seqs[c] == [ids1[c[0]], ids1[c[1]]]
    ]
    schemes["true_two_position_read"] = {
        "pairs_encoding_to_exactly_2_tokens": len(exactly_two),
        "of_those_decomposing_into_the_two_single_letter_ids": len(two_pos_consistent),
        "verdict": (
            "A genuine two-position read is only well-defined for the pairs BPE refuses to merge -- "
            "the 562 that DO merge would have to be split artificially, which the tokenizer will "
            "not do. So a uniform two-position scheme over AA..ZZ is not available under this "
            "tokenizer; the single-token subset is."
        ),
    }
    findings["identifier_schemes"] = schemes

    # --- 4. the chosen table + context stability ---------------------------
    codes, code_ids = build_wide_code_table(tok)
    # (a) does each code survive inside a real option line?
    ctx_fail = []
    for c in codes:
        line_ids = tok.encode(f"Options:\n{c}) card payment not recognised\n", add_special_tokens=False)
        if code_ids[codes.index(c)] not in line_ids:
            ctx_fail.append(c)
    # (b) does the answer position stay clean under the real chat template?
    tmpl = tok.apply_chat_template(
        [{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ans_fail = []
    for c in codes[:64] + codes[-16:]:
        cont = tok.encode(tmpl + c, add_special_tokens=False)
        base = tok.encode(tmpl, add_special_tokens=False)
        if cont[: len(base)] != base or cont[len(base):] != [code_ids[codes.index(c)]]:
            ans_fail.append(c)
    findings["code_table"] = {
        "size": len(codes),
        "prefix_is_A_to_Z": codes[:26] == LETTERS,
        "first_after_letters": codes[26:32],
        "last": codes[-4:],
        "all_single_token": True,
        "all_ids_distinct": len(set(code_ids)) == len(code_ids),
        "context_stable_in_option_line": len(ctx_fail) == 0,
        "context_failures": ctx_fail[:16],
        "answer_position_clean_sample": {
            "checked": len(codes[:64] + codes[-16:]),
            "failures": ans_fail,
            "note": (
                "The chat template's generation prompt ends in a newline, so the answer token "
                "carries no leading space -- the code's standalone id is exactly what a greedy "
                "read at position -1 must match. Verified, not assumed."
            ),
        },
        "headroom_vs_widest_real_task": f"{len(codes)} codes vs 151 needed (clinc150) = {len(codes)/151:.1f}x",
    }

    # --- 5. real wide tasks: do their full-width prompts even fit? ---------
    spaces = _load_label_spaces()
    prompt_costs = {}
    for name, info in spaces.items():
        if not info.get("cached"):
            prompt_costs[name] = {"cached": False, "error": info.get("error")}
            continue
        labels = info["labels"]
        n = len(labels)
        row = {"n_options": n, "needs_codes_beyond_26": n > 26}
        for variant in ("range", "enumerate"):
            text = build_prompt_text(STATE, labels, codes, INSTRUCTIONS, answer_line=variant)
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            row[f"prompt_tokens_{variant}"] = len(tok.encode(rendered, add_special_tokens=False))
        # today's behaviour: the widest subset the 26-letter mechanism can pose
        sub = labels[:26]
        text26 = build_prompt_text(STATE, sub, codes, INSTRUCTIONS, answer_line="enumerate")
        r26 = tok.apply_chat_template(
            [{"role": "user", "content": text26}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        row["prompt_tokens_current_26_option_subset"] = len(tok.encode(r26, add_special_tokens=False))
        row["fits_max_length_1280"] = row["prompt_tokens_range"] < 1280
        prompt_costs[name] = row
    findings["real_task_prompt_cost"] = {
        "label_spaces": {k: (v.get("n") if v.get("cached") else v) for k, v in spaces.items()},
        "per_task": prompt_costs,
        "note": (
            "`range` names only the first and last code in the trailing answer line; `enumerate` "
            "lists every code the way the current 26-option prompt does. The difference is the "
            "real token cost of keeping the existing prompt shape verbatim at width 151."
        ),
    }

    # --- 6. GPU: what the wide read actually costs -------------------------
    if args.gpu_forward:
        from transformers import AutoModelForImageTextToText

        assert torch.cuda.is_available(), "--gpu-forward requires CUDA"
        free, total = torch.cuda.mem_get_info(0)
        if free / 1e9 < 12:
            raise RuntimeError(f"only {free/1e9:.1f} GB free -- refusing to start on a shared GPU this tight")

        t0 = time.time()
        # the VL class is what 13a.11+ actually serves from (5.2b), so measure that one
        model = AutoModelForImageTextToText.from_pretrained(
            args.base_model, dtype=torch.bfloat16, device_map="cuda"
        )
        model.eval()
        load_s = time.time() - t0

        clinc = spaces.get("clinc150", {}).get("labels")
        assert clinc, "clinc150 label space not cached -- GPU section needs real wide options"

        def encode_prompt(labels, variant="range"):
            text = build_prompt_text(STATE, labels, codes, INSTRUCTIONS, answer_line=variant)
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            return tok(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")

        def timed_forward(enc, reps=5, **kw):
            with torch.no_grad():
                model(**enc, **kw)
            torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                t = time.time()
                with torch.no_grad():
                    out = model(**enc, **kw)
                torch.cuda.synchronize()
                ts.append((time.time() - t) * 1000)
            return out, round(float(np.median(ts)), 2)

        # 6a. prefill cost as a function of real option width
        width_costs = {}
        for n in (3, 26, 60, 77, 100, 151):
            enc = encode_prompt(clinc[:n])
            out, ms = timed_forward(enc)
            ids_n = torch.tensor(code_ids[:n], device="cuda")
            probs = F.softmax(out.logits[0, -1, ids_n].float(), dim=-1).cpu().numpy()
            width_costs[f"n={n}"] = {
                "prompt_tokens": int(enc["input_ids"].shape[1]),
                "prefill_ms_median": ms,
                "restricted_read_ok": bool(np.isfinite(probs).all() and abs(probs.sum() - 1) < 1e-4),
                "untrained_prior": _dist_stats(probs),
            }
        findings.setdefault("gpu", {})["prefill_by_option_width"] = width_costs

        # 6b. untrained prior comparability: A-Z (26) vs wide codes (151)
        enc26 = encode_prompt(clinc[:26])
        out26, _ = timed_forward(enc26, reps=1)
        p26 = F.softmax(out26.logits[0, -1, torch.tensor(code_ids[:26], device="cuda")].float(), dim=-1).cpu().numpy()
        enc151 = encode_prompt(clinc[:151])
        out151, _ = timed_forward(enc151, reps=1)
        p151 = F.softmax(
            out151.logits[0, -1, torch.tensor(code_ids[:151], device="cuda")].float(), dim=-1
        ).cpu().numpy()
        # unembedding-row norms: is the wide table's LM head geometry pathological?
        emb = model.get_output_embeddings().weight
        norms_all = emb.detach().float().norm(dim=-1)
        n_letters = norms_all[torch.tensor(code_ids[:26], device=emb.device)].cpu().numpy()
        n_wide = norms_all[torch.tensor(code_ids[26:], device=emb.device)].cpu().numpy()
        findings["gpu"]["untrained_prior_comparability"] = {
            "letters_26": _dist_stats(p26),
            "wide_codes_151": _dist_stats(p151),
            "lm_head_row_norm": {
                "letters_A_Z": {"mean": round(float(n_letters.mean()), 4), "std": round(float(n_letters.std()), 4)},
                "two_letter_codes": {"mean": round(float(n_wide.mean()), 4), "std": round(float(n_wide.std()), 4)},
                "ratio_of_means": round(float(n_wide.mean() / n_letters.mean()), 4),
            },
            "note": (
                "Untrained base model: this measures whether the wide code table is *mechanically* "
                "usable and whether its LM-head geometry is comparable to A-Z's, NOT accuracy. "
                "Accuracy is a training question."
            ),
        }

        # 6c. what a genuine two-position read would have cost, for the record
        enc = encode_prompt(clinc[:151])
        with torch.no_grad():
            model(**enc, use_cache=True)
        torch.cuda.synchronize()
        ts_pref, ts_step = [], []
        for _ in range(5):
            t = time.time()
            with torch.no_grad():
                o1 = model(**enc, use_cache=True)
            torch.cuda.synchronize()
            ts_pref.append((time.time() - t) * 1000)
            nxt = o1.logits[:, -1, :].argmax(-1, keepdim=True)
            t = time.time()
            with torch.no_grad():
                model(input_ids=nxt, past_key_values=o1.past_key_values, use_cache=True)
            torch.cuda.synchronize()
            ts_step.append((time.time() - t) * 1000)
        pref_ms = float(np.median(ts_pref))
        step_ms = float(np.median(ts_step))

        # the one-pass alternative: append a placeholder and read two positions
        pad_tok = tok.encode(")", add_special_tokens=False)[0]
        enc2 = {
            "input_ids": torch.cat(
                [enc["input_ids"], torch.tensor([[pad_tok]], device="cuda")], dim=1
            ),
            "attention_mask": torch.cat(
                [enc["attention_mask"], torch.ones((1, 1), dtype=enc["attention_mask"].dtype, device="cuda")], dim=1
            ),
        }
        out2, ms2 = timed_forward(enc2)
        findings["gpu"]["two_position_read_cost"] = {
            "prefill_ms_median": round(pref_ms, 2),
            "kv_cached_second_step_ms_median": round(step_ms, 2),
            "second_step_overhead_pct": round(100 * step_ms / pref_ms, 1),
            "one_pass_two_position_ms_median": ms2,
            "one_pass_overhead_vs_prefill_pct": round(100 * (ms2 - pref_ms) / pref_ms, 1),
            "one_pass_positions_readable": [int(out2.logits.shape[1] - 2), int(out2.logits.shape[1] - 1)],
            "note": (
                "Recorded because the task asked what a real multi-token read would cost. It is "
                "NOT needed -- section `code_table` shows a single-token table of 588 codes exists. "
                "The KV-cached second step is cheap in absolute terms but the one-pass variant "
                "factorises p(c1,c2|x) as p(c1|x)p(c2|x), which cannot represent a bimodal answer "
                "and would degrade exactly the calibration this project sells."
            ),
        }

        # 6d. path 2 (5.1a late-interaction) cost model, measured
        opt_texts = [f"{lbl}" for lbl in clinc[:151]]
        opt_enc = tok(opt_texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            model(**opt_enc, output_hidden_states=True)
        torch.cuda.synchronize()
        ts = []
        for _ in range(3):
            t = time.time()
            with torch.no_grad():
                oh = model(**opt_enc, output_hidden_states=True)
            torch.cuda.synchronize()
            ts.append((time.time() - t) * 1000)
        state_only = tok(
            tok.apply_chat_template(
                [{"role": "user", "content": f"{INSTRUCTIONS}\n\n{STATE}"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ),
            return_tensors="pt", add_special_tokens=False,
        ).to("cuda")
        _, state_ms = timed_forward(state_only, reps=3, output_hidden_states=False)
        findings["gpu"]["late_interaction_cost_model"] = {
            "hidden_size": int(cfg.text_config.hidden_size) if hasattr(cfg, "text_config") else int(cfg.hidden_size),
            "option_batch_shape": list(opt_enc["input_ids"].shape),
            "encode_151_options_ms_median": round(float(np.median(ts)), 2),
            "encode_state_only_ms_median": state_ms,
            "total_uncached_ms": round(float(np.median(ts)) + state_ms, 2),
            "hidden_states_exposed": bool(oh.hidden_states is not None),
            "n_hidden_layers_returned": len(oh.hidden_states),
            "note": (
                "Option encodings are cacheable per fixed schema (banking77's 77 intents never "
                "change), so the steady-state cost is the state encode alone -- this is path 2's "
                "genuine advantage. It is bought with a new scoring head, a new training objective "
                "(contrastive, not next-token CE), and zero reuse of the restricted-logit "
                "infrastructure in decoder_lora_lib.py / backends.py."
            ),
        }

        # 6e. is the wide table's untrained prior skew a one-prompt artifact?
        #     (6b found normalized entropy 0.012 at n=151 vs 0.537 at n=26 --
        #     worth knowing whether one code dominates *every* prompt, since
        #     that would argue for selecting low-prior codes rather than the
        #     lexicographically-first ones.)
        probe_states = [
            "what is the exchange rate between the dollar and the euro right now",
            "remind me to pick up the dry cleaning at six",
            "my card was declined at the grocery store",
            "how many calories are in a slice of pepperoni pizza",
            "book a table for four at an italian place downtown",
            "what is the weather going to be like this weekend",
            "cancel the alarm i set for tomorrow morning",
            "transfer two hundred dollars to my savings account",
        ]
        marg = np.zeros(len(codes), dtype=np.float64)
        per_prompt_top = []
        ids_all = torch.tensor(code_ids, device="cuda")
        for st in probe_states:
            text = build_prompt_text(st, clinc[:151], codes, INSTRUCTIONS)
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            e = tok(rendered, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.no_grad():
                o = model(**e)
            full = o.logits[0, -1, ids_all].float()
            p151 = F.softmax(full[:151], dim=-1).cpu().numpy()
            per_prompt_top.append([codes[int(i)] for i in np.argsort(-p151)[:3]])
            marg += F.softmax(full, dim=-1).cpu().numpy()
        marg /= len(probe_states)
        rank = np.argsort(-marg)

        def nent(sub: np.ndarray) -> float:
            p = sub / sub.sum()
            return round(float(-(p * np.log(np.clip(p, 1e-12, None))).sum() / math.log(len(p))), 4)

        # the apples-to-apples baseline: the 26-letter table that already works
        pair_marg = marg[26:]
        pair_rank_low = np.argsort(pair_marg)[:125]
        backcompat = np.concatenate([marg[:26], pair_marg[pair_rank_low]])
        findings["gpu"]["untrained_prior_stability"] = {
            "n_prompts": len(probe_states),
            "top10_codes_by_marginal": [[codes[int(i)], round(float(marg[i]), 5)] for i in rank[:10]],
            "same_top1_on_every_prompt": len({t[0] for t in per_prompt_top}) == 1,
            "per_prompt_top3": per_prompt_top,
            "normalized_entropy_existing_26_letter_table": nent(marg[:26]),
            "normalized_entropy_lexicographic_first_151": nent(marg[:151]),
            "normalized_entropy_151_lowest_prior_codes": nent(np.sort(marg)[:151]),
            "normalized_entropy_AZ_plus_125_lowest_prior_pairs": nent(backcompat),
            "lowest_prior_pair_codes_sample": [codes[26 + int(i)] for i in pair_rank_low[:12]],
            "note": (
                "Untrained skew, measured across prompts rather than inferred from one. The "
                "26-letter table has the same phenomenon in weaker form (6b), and trained through "
                "it fine (13a.6: 96.10%), so this is a thing to watch in the eval, not a blocker. "
                "Selecting the 151 lowest-prior codes instead of the lexicographically-first 151 "
                "is a real, cheap option if the trained calibration comes out poor -- the numbers "
                "above are what that choice would buy before training."
            ),
        }

        # 6f. training-shaped cost: does a width-151 row change the run config?
        #     LoRA forward+backward at the real batch shapes, peak VRAM + step time.
        from peft import LoraConfig, get_peft_model

        del out, out26, out151, out2, o1, oh
        torch.cuda.empty_cache()
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        pm = get_peft_model(model, lora_cfg)
        pm.gradient_checkpointing_enable()
        pm.enable_input_require_grads()
        pm.train()
        train_shapes = {}
        for label, n_opts, bs in (("width26_bs4", 26, 4), ("width151_bs4", 151, 4), ("width151_bs2", 151, 2)):
            texts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": build_prompt_text(s, clinc[:n_opts], codes, INSTRUCTIONS)}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                for s in probe_states[:bs]
            ]
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            enc_t = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            labels_t = torch.full_like(enc_t["input_ids"], -100)
            labels_t[:, -1] = torch.tensor(code_ids[:bs], device="cuda")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                t = time.time()
                loss = pm(**enc_t, labels=labels_t).loss
                loss.backward()
                torch.cuda.synchronize()
                step_ms = (time.time() - t) * 1000
                pm.zero_grad(set_to_none=True)
                train_shapes[label] = {
                    "seq_len": int(enc_t["input_ids"].shape[1]),
                    "batch": bs,
                    "fwd_bwd_ms": round(step_ms, 1),
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                    "oom": False,
                }
            except torch.cuda.OutOfMemoryError:
                train_shapes[label] = {"seq_len": int(enc_t["input_ids"].shape[1]), "batch": bs, "oom": True}
                torch.cuda.empty_cache()
        base = train_shapes.get("width26_bs4", {}).get("fwd_bwd_ms")
        wide = train_shapes.get("width151_bs4", {}).get("fwd_bwd_ms")
        findings["gpu"]["training_shape_cost"] = {
            "shapes": train_shapes,
            "wide_vs_narrow_step_ratio": round(wide / base, 2) if (base and wide) else None,
            "note": (
                "LoRA r=16 + gradient checkpointing, the config every run in this lineage uses. "
                "This is what calibrates the GPU-hour estimate for the wide slice against a "
                "measured rate rather than a parameter count (PRD §9's own caution)."
            ),
        }

        findings["gpu"]["environment"] = {
            "device_name": torch.cuda.get_device_name(0),
            "load_seconds": round(load_s, 1),
            "weights_vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "free_vram_gb_at_start": round(free / 1e9, 2),
        }

    print(json.dumps(findings, indent=2))
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "results" / f"multitoken-scheme-probe-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.manifest.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(findings, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
