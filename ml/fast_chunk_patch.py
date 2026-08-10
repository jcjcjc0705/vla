"""Make action chunking usable on a PNG-in-parquet LeRobotDataset.

    import fast_chunk_patch; fast_chunk_patch.apply()

⚠️ **Without this, training this dataset is ~70x slower than it needs to be**,
and nothing reports an error -- the GPU simply idles at 96%.

**What goes wrong.** ACT asks for ``action`` at 100 future steps
(``chunk_size=100``), which ``DatasetReader._query_hf_dataset`` serves with::

    torch.stack(self.hf_dataset[key][relative_indices])

Indexing 100 rows makes ``datasets`` *format* those rows, and formatting decodes
every column -- including the three PNG image columns. So each training sample
decodes **~300 images it then throws away** to obtain 100 six-float vectors.
Measured on 486 episodes / 100,096 frames::

    per sample        5.8 ms without chunking  ->  488 ms with it
    cProfile          99% in _query_hf_dataset, 3.86 s of PIL decode per 10 samples
    lerobot-train     updt_s 0.385   data_s 8.875     (batch 64, 4 workers)

**Why it does not bite everyone.** With ``use_videos=True`` the images live in
MP4 files, not in the Arrow table, so ``_query_hf_dataset`` skips them by name
(``video_keys``) and the row fetch is cheap. This project deliberately chose
``use_videos=False`` (PNG in parquet) to keep video decode *out* of the
dataloader -- and that is precisely the configuration that triggers this. The
choice is still right; it just needs this patch.

**The fix**: project the columns away before fetching the rows, so there is
nothing to decode. ``select_columns`` is cached per key -- it is a schema
operation, but doing it 100k times still costs more than keeping it.

    h[key][idx]                              476.40 ms
    h.select_columns([key]).select(idx)[key]   6.68 ms   <- identical tensors

Verified byte-identical (``torch.allclose`` plus shape and dtype) against the
unpatched path, and the images and other keys in the returned item are
untouched -- this only changes how the chunked columns are read.
"""
from __future__ import annotations

import torch

_applied = False


def apply() -> bool:
    """Patch ``DatasetReader._query_hf_dataset``. Idempotent; returns True once."""
    global _applied
    if _applied:
        return False

    from lerobot.datasets import dataset_reader as _dr

    def _query_hf_dataset(self, query_indices):
        cache = getattr(self, "_col_projection", None)
        if cache is None:
            cache = self._col_projection = {}
        out = {}
        for key, q in query_indices.items():
            if key in self._meta.video_keys:
                continue
            rel = (q if self._absolute_to_relative_idx is None
                   else [self._absolute_to_relative_idx[i] for i in q])
            proj = cache.get(key)
            if proj is None:
                proj = cache[key] = self.hf_dataset.select_columns([key])
            out[key] = torch.stack(proj.select(rel)[key][:])
        return out

    _dr.DatasetReader._query_hf_dataset = _query_hf_dataset
    _applied = True
    return True
