import collections
import json

from transformers import AutoTokenizer

from training.decoder_lora_lib import build_code_table

SOURCES = ('bjb:clinc150/intent', 'bjb:ledgar/provision_type',
           'bjb:banking77/intent', 'bjb:massive/intent')

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B')
codes, _ = build_code_table(tok)
lens = collections.defaultdict(list)
with open('/data/interns/studentiotlab/ekvachan/bench-repo/exports/full/public.jsonl') as f:
    for line in f:
        rec = json.loads(line)
        if rec['source'] not in SOURCES:
            continue
        opts = rec['options']
        n = len(opts)
        ol = '\n'.join(f'{codes[j]}) {opts[j]}' for j in range(n))
        if n <= 26:
            tail = "Answer with a single letter (%s)." % '/'.join(codes[:n])
        else:
            tail = ('Answer with a single option code from the list above '
                    f'({codes[0]} .. {codes[n-1]}).')
        text = f"{rec['instructions']}\n\n{rec['state']}\n\nOptions:\n{ol}\n\n{tail}"
        msg = [{'role': 'user', 'content': text}]
        rend = tok.apply_chat_template(msg, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
        lens[rec['source']].append(len(tok(rend)['input_ids']))
for src, v in lens.items():
    v.sort()
    print(src, 'n=', len(v), 'min=', v[0], 'p50=', v[len(v) // 2],
          'p90=', v[int(len(v) * 0.9)], 'p99=', v[int(len(v) * 0.99)],
          'max=', v[-1], flush=True)
