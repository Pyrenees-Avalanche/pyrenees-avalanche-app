import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import Sampler
from tqdm import tqdm


# ------------------------
# Base
# ------------------------
class CachedSamplerBase(Sampler):
    def __init__(self, dataset, patch_size, cache_prefix):
        cache_dir = Path("dataset/cache")
        cache_dir.mkdir(exist_ok=True, parents=True)

        self.pos_file = cache_dir / f"{cache_prefix}_pos_{patch_size}.npy"
        self.neg_file = cache_dir / f"{cache_prefix}_neg_{patch_size}.npy"
        self.map_file = cache_dir / f"{cache_prefix}_event_map_{patch_size}.pkl"

        self.dataset = dataset

        # Load or build caches
        if self.pos_file.exists() and self.neg_file.exists():
            self.positive_indices = np.load(self.pos_file).tolist()
            self.negative_indices = np.load(self.neg_file).tolist()
        else:
            self.positive_indices, self.negative_indices = [], []

            was_transform = getattr(dataset, "apply_transform", True)
            try:
                if hasattr(dataset, "apply_transform"):
                    dataset.apply_transform = False
                for i in tqdm(range(len(dataset)), desc="Scanning dataset for positives"):
                    sample = dataset[i]
                    if sample["mask"].any():
                        self.positive_indices.append(i)
                    else:
                        self.negative_indices.append(i)
            finally:
                if hasattr(dataset, "apply_transform"):
                    dataset.apply_transform = was_transform

            np.save(self.pos_file, np.asarray(self.positive_indices, dtype=np.int32))
            np.save(self.neg_file, np.asarray(self.negative_indices, dtype=np.int32))

        if not self.positive_indices:
            raise ValueError("No positive samples found in dataset!")


# ============================================================
# 1) Ratio-balanced sampler (positives : negatives)
#    Supports "fixed" mode to fully replace FixedBalancedSampler
# ============================================================
class BalancedPosNegSampler(CachedSamplerBase):
    def __init__(
        self,
        dataset,
        patch_size,
        ratio: float = 1.0,
        cache_prefix: str = "train",
        fixed: bool = False,
        fixed_file: Optional[str] = None,
        rng_seed: Optional[int] = None,
        duplicate_shortfall: bool = True,   # <--- NEW: default duplicate
    ):
        """
        ratio = positives : negatives.
          - ratio=2.0 -> 2 positives per 1 negative (neg ≈ pos / 2)
          - ratio=1.0 -> 1 positive per 1 negative (neg ≈ pos)
          - ratio=0.5 -> 1 positive per 2 negatives (neg ≈ 2 * pos)
        fixed=True  -> select negatives once and reuse every epoch (like a fixed sampler).
        fixed_file  -> optional on-disk cache for the fixed selection.
        rng_seed    -> reproducible selection.
        duplicate_shortfall -> if True, when negatives are insufficient, pad by
                               sampling with replacement; if False, cap at available.
        """
        super().__init__(dataset, patch_size, cache_prefix=cache_prefix)
        self.ratio = float(max(1e-8, ratio))  # avoid divide-by-zero
        self.fixed = bool(fixed)
        self.fixed_file = Path(fixed_file) if fixed_file else None
        self.rng = np.random.default_rng(rng_seed)
        self.duplicate_shortfall = bool(duplicate_shortfall)

        # Build event → neg map (cached)
        if Path(self.map_file).exists():
            with open(self.map_file, "rb") as f:
                self.event_to_neg_indices = pickle.load(f)
        else:
            tmp = defaultdict(list)
            was_transform = getattr(dataset, "apply_transform", True)
            try:
                if hasattr(dataset, "apply_transform"):
                    dataset.apply_transform = False
                for idx in tqdm(self.negative_indices, desc="Grouping negatives by event"):
                    e = dataset[idx]["event"]
                    tmp[e].append(idx)
            finally:
                if hasattr(dataset, "apply_transform"):
                    dataset.apply_transform = was_transform
            self.event_to_neg_indices = {e: np.asarray(v, dtype=np.int32) for e, v in tmp.items()}
            with open(self.map_file, "wb") as f:
                pickle.dump(self.event_to_neg_indices, f)

        # Build event → pos map (in-memory)
        pos_tmp = defaultdict(list)
        for idx in self.positive_indices:
            e = dataset[idx]["event"]
            pos_tmp[e].append(idx)
        self.event_to_pos_indices = {e: np.asarray(v, dtype=np.int32) for e, v in pos_tmp.items()}

        # Events that have both positives and negatives
        self.events_with_pos = [e for e in self.event_to_pos_indices if e in self.event_to_neg_indices]

        # Fixed-mode storage
        self._fixed_negatives = None
        self._last_epoch_len = None

        if self.fixed:
            self._init_fixed_selection()

    # ---- helpers ------------------------------------------------------------
    def _pick_baseline_negatives(self, target_neg: int) -> np.ndarray:
        baseline_neg = np.empty((0,), dtype=np.int32)
        if self.events_with_pos and target_neg > 0:
            ne = len(self.events_with_pos)
            negs_per_event = target_neg // ne
            remainder = target_neg % ne
            picks = []
            for i, e in enumerate(self.events_with_pos):
                pool = self.event_to_neg_indices[e]
                n = negs_per_event + (1 if i < remainder else 0)
                if n <= 0:
                    continue
                n = min(n, len(pool))
                if n > 0:
                    picks.append(self.rng.choice(pool, n, replace=False))
            if picks:
                baseline_neg = np.concatenate(picks)
        return baseline_neg

    def _pad_with_replacement(self, chosen: np.ndarray, target: int) -> np.ndarray:
        """Pad to 'target' by sampling with replacement from negatives."""
        if not self.duplicate_shortfall or target <= len(chosen):
            return chosen
        need = target - len(chosen)
        if need <= 0 or len(self.negative_indices) == 0:
            return chosen
        # Prefer not-yet-used negatives first (no replacement), then replace if still short.
        remaining = np.setdiff1d(np.asarray(self.negative_indices, dtype=np.int32),
                                 chosen, assume_unique=False)
        take = min(need, remaining.size)
        parts = [chosen]
        if take > 0:
            parts.append(self.rng.choice(remaining, take, replace=False))
        still = need - take
        if still > 0:
            parts.append(self.rng.choice(self.negative_indices, still, replace=True))
        return np.concatenate(parts)

    def _init_fixed_selection(self):
        # Load negatives if cached on disk
        if self.fixed_file and self.fixed_file.exists():
            self._fixed_negatives = np.load(self.fixed_file).astype(np.int32)
            return

        pos_arr = np.asarray(self.positive_indices, dtype=np.int32)
        target_neg = int(round(len(pos_arr) / self.ratio))
        baseline_neg = self._pick_baseline_negatives(target_neg)

        # Honor shortfall policy also in fixed mode
        if self.duplicate_shortfall:
            baseline_neg = self._pad_with_replacement(baseline_neg, target_neg)
        # else cap as-is

        self._fixed_negatives = baseline_neg
        if self.fixed_file:
            self.fixed_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.fixed_file, self._fixed_negatives)

    # ---- Sampler API --------------------------------------------------------
    def __iter__(self):
        pos_arr = np.asarray(self.positive_indices, dtype=np.int32)

        if self.fixed:
            combined = np.concatenate([pos_arr, self._fixed_negatives])
            self._last_epoch_len = len(combined)
            idxs = combined.copy()
            self.rng.shuffle(idxs)  # shuffle order each epoch, set stays fixed
            return iter(idxs.tolist())

        # Non-fixed: new negatives each epoch
        target_neg = int(round(len(pos_arr) / self.ratio))
        baseline_neg = self._pick_baseline_negatives(target_neg)

        # Apply shortfall policy
        if self.duplicate_shortfall:
            baseline_neg = self._pad_with_replacement(baseline_neg, target_neg)
        # else cap as-is

        combined = np.concatenate([pos_arr, baseline_neg])
        self.rng.shuffle(combined)
        self._last_epoch_len = len(combined)
        return iter(combined.tolist())

    def __len__(self):
        if self._last_epoch_len is not None:
            return self._last_epoch_len
        if self.fixed and self._fixed_negatives is not None:
            return len(self.positive_indices) + len(self._fixed_negatives)
        # Estimate for non-fixed path
        target_neg = int(round(len(self.positive_indices) / self.ratio))
        if self.duplicate_shortfall:
            est_neg = target_neg
        else:
            # cap by what's pickable across events with positives
            # (rough lower bound = all negatives in those events)
            cap = sum(len(self.event_to_neg_indices[e]) for e in self.events_with_pos)
            est_neg = min(target_neg, cap)
        return len(self.positive_indices) + est_neg


# =====================================================================
# 2) Ratio-balanced + extra negatives from events with NO positives
#    Also supports fixed mode
# =====================================================================
class BalancedPosNegWithNoPosSampler(BalancedPosNegSampler):
    def __init__(
        self,
        dataset,
        patch_size,
        ratio: float = 1.0,
        no_pos_fraction: float = 0.0,
        cache_prefix: str = "train",
        fixed: bool = False,
        fixed_file: Optional[str] = None,
        rng_seed: Optional[int] = None,
        duplicate_shortfall: bool = True,   # <--- propagate policy
    ):
        """
        Extends the ratio-balanced sampler by adding extra negatives from events
        that have NO positives.

        - ratio: positives : negatives for events WITH positives.
        - no_pos_fraction ∈ [0, 1]: extra negatives drawn from events WITHOUT positives,
          equal to ceil(no_pos_fraction * BASELINE_SIZE), where
          BASELINE_SIZE = (#pos + #neg from events with positives).
        - fixed: if True, both baseline and extra negatives are picked once and reused each epoch.
        - duplicate_shortfall: pad (duplicate) to target when pools are small; if False, cap.
        """
        super().__init__(
            dataset,
            patch_size,
            ratio=ratio,
            cache_prefix=cache_prefix,
            fixed=False,           # delay fixing until we know no-pos pool
            fixed_file=None,
            rng_seed=rng_seed,
            duplicate_shortfall=duplicate_shortfall,
        )
        self.no_pos_fraction = float(np.clip(no_pos_fraction, 0.0, 1.0))
        self.fixed = bool(fixed)
        self.fixed_file = Path(fixed_file) if fixed_file else None
        self.rng = np.random.default_rng(rng_seed)

        # Events with no positives
        events_no_pos = [e for e in self.event_to_neg_indices if e not in self.event_to_pos_indices]
        self.events_no_pos = events_no_pos
        self.union_no_pos_negs = (
            np.concatenate([self.event_to_neg_indices[e] for e in events_no_pos])
            if events_no_pos else np.empty((0,), dtype=np.int32)
        )

        self._fixed_baseline = None
        self._fixed_extra = None
        self._last_epoch_len = None

        if self.fixed:
            self._init_fixed_selection()

    # ---- helpers ------------------------------------------------------------
    def _pick_extra_negatives(self, baseline_neg: np.ndarray, num_pos: int) -> np.ndarray:
        if self.no_pos_fraction <= 0.0 or self.union_no_pos_negs.size == 0:
            return np.empty((0,), dtype=np.int32)
        available = (np.setdiff1d(self.union_no_pos_negs, baseline_neg, assume_unique=False)
                     if baseline_neg.size > 0 else self.union_no_pos_negs)
        if available.size == 0:
            return np.empty((0,), dtype=np.int32)

        baseline_size = num_pos + len(baseline_neg)
        target_extra = int(np.ceil(self.no_pos_fraction * baseline_size))
        k = min(target_extra, available.size)
        picks = self.rng.choice(available, k, replace=False) if k > 0 else np.empty((0,), dtype=np.int32)

        # Optionally duplicate to reach exact target_extra
        if self.duplicate_shortfall and k < target_extra:
            need = target_extra - k
            # sample with replacement from the union pool
            dup = self.rng.choice(self.union_no_pos_negs, need, replace=True) if need > 0 else np.empty((0,), dtype=np.int32)
            picks = np.concatenate([picks, dup])

        return picks

    def _init_fixed_selection(self):
        # If cached fixed selection exists, load it (all negatives saved together)
        if self.fixed_file and self.fixed_file.exists():
            arr = np.load(self.fixed_file).astype(np.int32)
            self._fixed_baseline = arr
            self._fixed_extra = np.empty((0,), dtype=np.int32)
            return

        pos_arr = np.asarray(self.positive_indices, dtype=np.int32)
        target_neg = int(round(len(pos_arr) / self.ratio))
        baseline_neg = self._pick_baseline_negatives(target_neg)
        if self.duplicate_shortfall:
            baseline_neg = self._pad_with_replacement(baseline_neg, target_neg)

        extra_neg = self._pick_extra_negatives(baseline_neg, len(pos_arr))

        self._fixed_baseline = baseline_neg
        self._fixed_extra = extra_neg

        if self.fixed_file:
            self.fixed_file.parent.mkdir(parents=True, exist_ok=True)
            to_save = np.concatenate([baseline_neg, extra_neg])
            np.save(self.fixed_file, to_save)

    # ---- Sampler API --------------------------------------------------------
    def __iter__(self):
        pos_arr = np.asarray(self.positive_indices, dtype=np.int32)

        if self.fixed:
            combined = np.concatenate([pos_arr, self._fixed_baseline, self._fixed_extra])
            self._last_epoch_len = len(combined)
            idxs = combined.copy()
            self.rng.shuffle(idxs)
            return iter(idxs.tolist())

        # Non-fixed: sample each epoch
        target_neg = int(round(len(pos_arr) / self.ratio))
        baseline_neg = self._pick_baseline_negatives(target_neg)
        if self.duplicate_shortfall:
            baseline_neg = self._pad_with_replacement(baseline_neg, target_neg)

        extra_neg = self._pick_extra_negatives(baseline_neg, len(pos_arr))

        combined = np.concatenate([pos_arr, baseline_neg, extra_neg])
        self.rng.shuffle(combined)
        self._last_epoch_len = len(combined)
        return iter(combined.tolist())

    def __len__(self):
        if self._last_epoch_len is not None:
            return self._last_epoch_len
        if self.fixed and (self._fixed_baseline is not None):
            return len(self.positive_indices) + len(self._fixed_baseline) + (len(self._fixed_extra) if self._fixed_extra is not None else 0)

        # Estimate non-fixed
        pos_n = len(self.positive_indices)
        target_neg = int(round(pos_n / self.ratio))
        if self.duplicate_shortfall:
            baseline_est = pos_n + target_neg
        else:
            cap = sum(len(self.event_to_neg_indices[e]) for e in self.events_with_pos)
            baseline_est = pos_n + min(target_neg, cap)

        extra_est = int(np.ceil(self.no_pos_fraction * baseline_est))
        return baseline_est + extra_est