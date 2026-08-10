"""``lerobot-train`` with the chunked-action dataloader fix applied.

    python3 ml/train.py --dataset.repo_id=screamlab/omx_pick_cube \
        --dataset.root=data/lerobot_3cam --policy.type=act \
        --policy.push_to_hub=false --output_dir=outputs/act_v1 \
        --batch_size=64 --steps=60000 --num_workers=8 --wandb.enable=false

**Use this instead of the ``lerobot-train`` CLI for this dataset.** Every flag is
passed straight through -- this file adds no arguments and changes no defaults.
The only difference is the import above ``main()``.

Why it has to exist: see ``fast_chunk_patch``. Short version -- ACT's 100-step
action chunk makes lerobot decode ~300 PNGs per sample and discard them. On this
dataset, measured at batch 64 / 4 workers::

    lerobot-train        updt_s 0.441   data_s 9.575   smp/s   6
    ml/train.py          updt_s 0.172   data_s 0.012   smp/s 348

60k steps: about **7 days** versus about **3 hours**. The bare CLI still works
and reports no error -- the GPU just idles at 96%.

⚠️ **This forces ``dataloader_multiprocessing_context=fork``**, and that is not
cosmetic. lerobot defaults it to ``spawn``, and a spawned worker re-imports
everything from scratch, so it never sees a patch applied in this process. The
first attempt at this fix looked like it did nothing for exactly that reason:
``ds[i]`` was 49x faster in the parent while ``data_s`` did not move at all,
because none of the work happens in the parent. Pass the flag yourself to
override; a patched parent with spawned workers is the one combination that
silently wastes the whole speedup.

⚠️ ``--policy.push_to_hub=false`` is still required (lerobot rejects the config
without it, complaining about ``repo_id`` in a message that sounds unrelated).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fast_chunk_patch  # noqa: E402

_CTX = "--dataloader_multiprocessing_context"

if __name__ == "__main__":
    fast_chunk_patch.apply()
    argv = sys.argv[1:]
    if not any(a == _CTX or a.startswith(_CTX + "=") for a in argv):
        # fork, so the workers inherit the patched class. See the warning above.
        sys.argv = [sys.argv[0], f"{_CTX}=fork", *argv]
    from lerobot.scripts.lerobot_train import main

    sys.exit(main())
