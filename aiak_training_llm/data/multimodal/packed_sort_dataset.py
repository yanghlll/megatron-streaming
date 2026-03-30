"""PackedSeparateSortDataset — 分离 packed / non-packed 样本独立排序.

Non-packed 样本使用 full causal attention，计算量与 seq_len² 成正比，
长度差异对跨 rank 同步影响极大，需要严格按长度排序。
Packed 样本使用 block-diagonal attention，计算量更均匀，
排序仍有帮助但不如 non-packed 关键。

混合排序会导致等长但计算量迥异的 packed / non-packed 样本交错出现，
同一时刻不同 rank 可能一个在处理 expensive non-packed、另一个在处理
cheap packed，造成等待。

本类将 pool 内样本分为 packed / non-packed 两组，各自独立排序 + shuffle，
然后按原始比例交错合并输出（既保持时间均匀性，又使相邻同类样本尽量连续）。

判据：样本若拥有 cu_lengths 且包含多个子样本边界 (numel > 2)，则视为 packed。
"""

import random
from collections.abc import Callable, Iterator
from typing import TypeVar

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig


T_sample = TypeVar("T_sample")


def _is_packed(sample: T_sample) -> bool:
    """判断样本是否为离线 packed 样本.

    Packed 样本的 cu_lengths 有多个子样本边界 (numel > 2).
    Non-packed 样本没有 cu_lengths，或 cu_lengths 只有 [0, total_len] (numel <= 2).
    """
    cu = getattr(sample, "cu_lengths", None)
    if cu is None:
        return False
    length = cu.numel() if hasattr(cu, "numel") else len(cu)
    return length > 2


class PackedSeparateSortDataset(SavableDataset[T_sample]):
    """局部池化长度排序，packed / non-packed 样本分离独立排序.

    流程（每个 pool）:
      1. 累积 pool_size 个样本
      2. 按 _is_packed() 分成两组
      3. 各组独立按 key_fn 排序
      4. 各组独立用确定性种子 shuffle（保持跨 rank 一致性）
      5. 按原始比例交错合并输出（保持时间均匀性）
    """

    def __init__(
        self,
        dataset: SavableDataset[T_sample],
        *,
        pool_size: int,
        key_fn: Callable[[T_sample], int],
        ascending: bool,
        worker_config: WorkerConfig,
        tail_shuffle: bool = True,
        shuffle_seed: int | None = None,
        warmup_steps: int = 0,
        initial_pool_size: int = 10,
    ):
        super().__init__(worker_config=worker_config)
        assert pool_size > 0
        self.dataset = dataset
        self.pool_size = pool_size
        self.key_fn = key_fn
        self.ascending = ascending
        self.tail_shuffle = tail_shuffle
        base_seed = (
            shuffle_seed
            if shuffle_seed is not None
            else getattr(worker_config, "global_seed", 1234)
        )
        self._rng = random.Random(base_seed)

        self.warmup_steps = warmup_steps
        self.initial_pool_size = min(initial_pool_size, pool_size)

    def _get_current_pool_size(self, pool_flush_count: int) -> int:
        """Calculate current pool size with warmup."""
        if self.warmup_steps == 0 or pool_flush_count >= self.warmup_steps:
            return self.pool_size
        progress = pool_flush_count / self.warmup_steps
        return self.initial_pool_size + int((self.pool_size - self.initial_pool_size) * progress)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self) -> Iterator[T_sample]:
        pool: list[T_sample] = []
        pool_flush_count = 0
        for batch_idx, sample in enumerate(self.dataset):
            pool.append(sample)
            current_pool_size = self._get_current_pool_size(pool_flush_count)
            if len(pool) >= current_pool_size:
                yield from self._flush_pool(pool, batch_idx)
                pool.clear()
                pool_flush_count += 1
        if pool:
            yield from self._flush_tail(pool)
            pool.clear()

    # ---- flush / interleave helpers ----

    @staticmethod
    def _interleave(
        a: list[T_sample], b: list[T_sample],
    ) -> Iterator[T_sample]:
        """按原始比例交错合并两个已排序列表.

        假设 a 占多数、b 占少数。每输出 ceil(len(a)/len(b)) 个 a 样本后，
        插入 1 个 b 样本。如果其中一组为空，直接输出另一组。
        """
        if not b:
            yield from a
            return
        if not a:
            yield from b
            return

        # 让 a 始终是较多的一组
        if len(a) < len(b):
            a, b = b, a

        # 每输出 step 个 a 样本后插入 1 个 b 样本
        step = len(a) / len(b)  # >= 1.0
        ai, bi = 0, 0
        next_insert = step  # b 样本的下一个插入点
        for i in range(len(a) + len(b)):
            if bi < len(b) and (ai >= len(a) or ai >= next_insert):
                yield b[bi]
                bi += 1
                next_insert += step
            else:
                yield a[ai]
                ai += 1

    def _sort_and_shuffle(
        self, group: list[T_sample], seed: int,
    ) -> list[T_sample]:
        """排序 + 确定性 shuffle，保持跨 rank 一致性."""
        group.sort(key=self.key_fn, reverse=not self.ascending)
        random.Random(seed).shuffle(group)
        return group

    def _flush_pool(
        self, pool: list[T_sample], batch_idx: int,
    ) -> Iterator[T_sample]:
        """分离 packed/non-packed → 各自排序+shuffle → 按比例交错输出."""
        packed = [s for s in pool if _is_packed(s)]
        unpacked = [s for s in pool if not _is_packed(s)]

        seed_base = 42 + batch_idx
        if unpacked:
            self._sort_and_shuffle(unpacked, seed_base)
        if packed:
            self._sort_and_shuffle(packed, seed_base + 1_000_000)

        yield from self._interleave(unpacked, packed)

    def _flush_tail(self, pool: list[T_sample]) -> Iterator[T_sample]:
        """处理不足 pool_size 的尾部."""
        packed = [s for s in pool if _is_packed(s)]
        unpacked = [s for s in pool if not _is_packed(s)]

        unpacked.sort(key=self.key_fn, reverse=not self.ascending)
        packed.sort(key=self.key_fn, reverse=not self.ascending)

        if self.tail_shuffle:
            self._rng.shuffle(unpacked)
            self._rng.shuffle(packed)

        yield from self._interleave(unpacked, packed)

    # ---- SavableDataset 抽象方法委托 ----

    def worker_has_samples(self) -> bool:
        return self.dataset.worker_has_samples()

    def can_restore_sample(self) -> bool:
        return self.dataset.can_restore_sample()

    def assert_can_restore(self) -> None:
        self.dataset.assert_can_restore()

    def restore_sample(self, index):
        return self.dataset.restore_sample(index)

    def save_state(self):
        return self.dataset.save_state()

    def merge_states(self, states):
        return self.dataset.merge_states(states)

    def restore_state(self, state):
        self.dataset.restore_state(state)

    def config(self):
        return {
            "type": type(self).__qualname__,
            "pool_size": self.pool_size,
            "ascending": self.ascending,
            "tail_shuffle": self.tail_shuffle,
            "warmup_steps": self.warmup_steps,
            "initial_pool_size": self.initial_pool_size,
            "dataset": self.dataset.config(),
        }
