"""Prompt datasets and the distributed samplers used for RLVVR training."""
import json
import math
import os

import torch
from torch.utils.data import Dataset, Sampler


class TextPromptDataset(Dataset):
    """One prompt per line of `{dataset}/{split}.txt`."""

    def __init__(self, dataset, split="train"):
        with open(os.path.join(dataset, f"{split}.txt"), "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]
        self.metadatas = [{} for _ in self.prompts]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        return [e["prompt"] for e in examples], [e["metadata"] for e in examples]


class MetadataPromptDataset(Dataset):
    """One JSON row per line of `{dataset}/{split}_metadata.jsonl`; each row has a `prompt`."""

    def __init__(self, dataset, split="train"):
        with open(os.path.join(dataset, f"{split}_metadata.jsonl"), "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
        self.prompts = [row["prompt"] for row in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        return [e["prompt"] for e in examples], [e["metadata"] for e in examples]


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0, rollout_batches=1):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        self.rollout_batches = max(1, int(rollout_batches))
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def _permutation_for_cycle(self, population_size, cycle):
        g = torch.Generator()
        g.manual_seed(self.seed + cycle)
        return torch.randperm(population_size, generator=g).tolist(), g

    def _selected_unique_indices(self, population):
        """Return the unique prompt rows for this sampler step.

        Training checks prompt uniqueness over the whole rollout, not only one
        per-rank batch.  When a rollout crosses the boundary between two
        shuffled passes through a small active pool, taking the tail of one
        permutation plus the head of the next can repeat a row inside the same
        rollout.  Build the rollout-sized block first and filter the next
        permutation against the tail before slicing this sampler step.
        """
        population_size = len(population)
        rollout_m = self.m * self.rollout_batches
        if rollout_m > population_size:
            raise ValueError(
                "DistributedKRepeatSampler rollout requires more unique samples "
                f"than the active population provides: rollout_unique={rollout_m} "
                f"population={population_size}"
            )

        rollout_epoch = self.epoch // self.rollout_batches
        batch_in_rollout = self.epoch % self.rollout_batches
        rollout_cycle_len = max(1, math.ceil(population_size / rollout_m))
        cycle = rollout_epoch // rollout_cycle_len
        offset = (rollout_epoch * rollout_m) % population_size

        order, g = self._permutation_for_cycle(population_size, cycle)
        if offset + rollout_m <= population_size:
            block = order[offset:offset + rollout_m]
        else:
            remaining = population_size - offset
            block = order[offset:]
            used = set(block)
            next_cycle = cycle + 1
            while len(block) < rollout_m:
                next_order, _ = self._permutation_for_cycle(population_size, next_cycle)
                block.extend(idx for idx in next_order if idx not in used)
                block = block[:rollout_m]
                used = set(block)
                next_cycle += 1

        start = batch_in_rollout * self.m
        selected = block[start:start + self.m]
        return [population[idx] for idx in selected], g

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            population = list(range(len(self.dataset)))

            indices, g = self._selected_unique_indices(population)

            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs


class DistributedKRepeatSourceBalancedSampler(DistributedKRepeatSampler):
    """K-repeat sampler with an exact deterministic source mix per batch."""

    def __init__(
        self,
        dataset,
        batch_size,
        k,
        num_replicas,
        rank,
        source_indices,
        source_weights,
        seed=0,
        rollout_batches=1,
        rotate_remainder=False,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            k=k,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
            rollout_batches=rollout_batches,
        )
        self.source_names = tuple(source_weights)
        if not self.source_names:
            raise ValueError("source_weights must be non-empty")
        self.source_indices = {
            source: [int(index) for index in source_indices.get(source, [])]
            for source in self.source_names
        }
        missing = [source for source, indices in self.source_indices.items() if not indices]
        if missing:
            raise ValueError(f"source-balanced sampler has empty sources: {missing}")
        weights = {source: float(source_weights[source]) for source in self.source_names}
        if any(weight <= 0 for weight in weights.values()):
            raise ValueError(f"source weights must be positive: {weights}")
        weight_sum = sum(weights.values())
        exact_counts = {source: self.m * weights[source] / weight_sum for source in self.source_names}
        self._source_base_counts = {
            source: int(math.floor(exact_counts[source])) for source in self.source_names
        }
        self._source_remainder = self.m - sum(self._source_base_counts.values())
        self._remainder_order = tuple(sorted(
            self.source_names,
            key=lambda source: (-(exact_counts[source] % 1.0), self.source_names.index(source)),
        ))
        self.rotate_remainder = bool(rotate_remainder)
        self.source_counts_per_batch = self._counts_for_epoch(0)
        if any(count <= 0 for count in self.source_counts_per_batch.values()):
            raise ValueError(
                "global unique prompts per batch are too small for the requested sources: "
                f"m={self.m} counts={self.source_counts_per_batch}"
            )
        for source, count in self._source_base_counts.items():
            rollout_count = count * self.rollout_batches
            if self.rotate_remainder and self._source_remainder:
                rollout_count += math.ceil(
                    self.rollout_batches * self._source_remainder / len(self.source_names)
                )
            elif source in self._remainder_order[:self._source_remainder]:
                rollout_count += self.rollout_batches
            if rollout_count > len(self.source_indices[source]):
                raise ValueError(
                    f"source {source} needs {rollout_count} unique rows per rollout epoch, "
                    f"but only {len(self.source_indices[source])} are available"
                )

    def _counts_for_epoch(self, epoch):
        counts = dict(self._source_base_counts)
        if not self._source_remainder:
            return counts
        if self.rotate_remainder:
            start = (int(epoch) * self._source_remainder) % len(self.source_names)
            extra_sources = [
                self._remainder_order[(start + offset) % len(self.source_names)]
                for offset in range(self._source_remainder)
            ]
        else:
            extra_sources = self._remainder_order[:self._source_remainder]
        for source in extra_sources:
            counts[source] += 1
        return counts

    def _source_draw_offset(self, source, epoch):
        base = int(epoch) * self._source_base_counts[source]
        if not self.rotate_remainder or not self._source_remainder:
            extra = int(epoch) if source in self._remainder_order[:self._source_remainder] else 0
            return base + extra
        source_position = self._remainder_order.index(source)
        complete_cycles, partial = divmod(
            int(epoch) * self._source_remainder,
            len(self.source_names),
        )
        return base + complete_cycles + int(source_position < partial)

    def _source_rows_at_offset(self, source, count, offset):
        population = self.source_indices[source]
        population_size = len(population)
        cycle, position = divmod(offset, population_size)
        selected = []
        used = set()
        source_position = self.source_names.index(source)
        while len(selected) < count:
            order = self._source_permutation(source_position, population_size, cycle)
            for local_index in order[position:]:
                item = population[local_index]
                if item not in used:
                    selected.append(item)
                    used.add(item)
                    if len(selected) == count:
                        break
            cycle += 1
            position = 0
        return selected

    def _source_permutation(self, source_position, population_size, cycle):
        generator = torch.Generator()
        generator.manual_seed(self.seed + (source_position + 1) * 1_000_003 + cycle)
        return torch.randperm(population_size, generator=generator).tolist()

    def _source_rollout_block(self, source, count_per_batch, rollout_epoch):
        population = self.source_indices[source]
        population_size = len(population)
        block_size = count_per_batch * self.rollout_batches
        source_position = self.source_names.index(source)
        cycle_length = max(1, math.ceil(population_size / block_size))
        cycle = rollout_epoch // cycle_length
        offset = (rollout_epoch * block_size) % population_size
        order = self._source_permutation(source_position, population_size, cycle)
        if offset + block_size <= population_size:
            block = order[offset:offset + block_size]
        else:
            block = order[offset:]
            used = set(block)
            next_cycle = cycle + 1
            while len(block) < block_size:
                next_order = self._source_permutation(source_position, population_size, next_cycle)
                block.extend(index for index in next_order if index not in used)
                block = block[:block_size]
                used = set(block)
                next_cycle += 1
        return [population[index] for index in block]

    def _selected_source_indices(self):
        if self.rotate_remainder:
            counts = self._counts_for_epoch(self.epoch)
            indices = []
            for source in self.source_names:
                indices.extend(
                    self._source_rows_at_offset(
                        source,
                        counts[source],
                        self._source_draw_offset(source, self.epoch),
                    )
                )
            if len(indices) != self.m or len(set(indices)) != self.m:
                raise RuntimeError(
                    f"invalid rotating source-balanced prompt block: total={len(indices)} "
                    f"unique={len(set(indices))} expected={self.m}"
                )
            return indices

        rollout_epoch = self.epoch // self.rollout_batches
        batch_in_rollout = self.epoch % self.rollout_batches
        indices = []
        for source in self.source_names:
            count = self.source_counts_per_batch[source]
            block = self._source_rollout_block(source, count, rollout_epoch)
            start = batch_in_rollout * count
            indices.extend(block[start:start + count])
        if len(indices) != self.m or len(set(indices)) != self.m:
            raise RuntimeError(
                f"invalid source-balanced prompt block: total={len(indices)} "
                f"unique={len(set(indices))} expected={self.m}"
            )
        return indices

    def _global_repeated_indices(self, indices):
        repeated_indices = [index for index in indices for _ in range(self.k)]
        generator = torch.Generator()
        generator.manual_seed(self.seed + 10_000_019 + self.epoch)
        order = torch.randperm(len(repeated_indices), generator=generator).tolist()
        return [repeated_indices[index] for index in order]

    def __iter__(self):
        while True:
            indices = self._selected_source_indices()
            shuffled = self._global_repeated_indices(indices)
            start = self.rank * self.batch_size
            yield shuffled[start:start + self.batch_size]
