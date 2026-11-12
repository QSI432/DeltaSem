"""Reference implementation of anchored certificate replay.

This module implements the exact algebra used in Sections IV--VI. It has no
model server, relational optimizer, or hardware performance measurements.
Every interval supplied to an Anchor or SignedSlice is an assumed valid
certificate. Probability allocation and construction of actual audit intervals
are implemented separately in certificate_reference.py.
"""
from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping


@dataclass(frozen=True)
class Occurrence:
    identity: str
    semantic_key: str
    weight: float
    group: str = "group"
    preimage: str = ""

    def __post_init__(self):
        if not self.identity or not self.semantic_key:
            raise ValueError("identities must be nonempty")
        if not isfinite(self.weight) or self.weight < 0:
            raise ValueError("weights must be finite and nonnegative")


class LabelLedger:
    """Immutable first-write labels indexed by canonical semantic input."""
    def __init__(self):
        self.labels: dict[str, int] = {}

    def observe(self, semantic_key: str, label: int):
        if label not in (0, 1):
            raise ValueError("the reference predicate is binary")
        if semantic_key in self.labels and self.labels[semantic_key] != label:
            raise ValueError("reference identity produced conflicting labels")
        self.labels[semantic_key] = label


class Snapshot:
    """Occurrence identities are immutable, even after their deletion."""
    def __init__(self, occurrences: Iterable[Occurrence] = ()):
        self.known_occurrences: dict[str, Occurrence] = {}
        self.active: dict[str, Occurrence] = {}
        for occurrence in occurrences:
            self.insert(occurrence)

    def insert(self, occurrence: Occurrence):
        previous = self.known_occurrences.get(occurrence.identity)
        if previous is not None and previous != occurrence:
            raise ValueError("an occurrence identity cannot change its contents")
        self.known_occurrences[occurrence.identity] = occurrence
        self.active[occurrence.identity] = occurrence

    def delete(self, identity: str):
        self.active.pop(identity)

    def residual(self, ledger: LabelLedger, group: str):
        return {
            key: value for key, value in self.active.items()
            if value.group == group and value.semantic_key not in ledger.labels
        }

    def exact_total(self, ledger: LabelLedger, group: str):
        return sum(
            value.weight * ledger.labels[value.semantic_key]
            for value in self.active.values()
            if value.group == group and value.semantic_key in ledger.labels
        )


class IncompatibleSliceFrame(ValueError):
    """A frozen signed slice no longer belongs to its original net-delta side."""


@dataclass(frozen=True)
class SignedSlice:
    """Certified total of one frozen, disjoint insertion or retraction slice.

    All identities must still be on the corresponding net-delta side at replay.
    For example, partial reinsertion invalidates a whole-deletion-slice shortcut.
    Drop that shortcut and replay raw bounded leaves, or transport its interval
    with a separately retained slice anchor. Anchor.project rejects this case
    explicitly instead of silently applying a certificate to a smaller slice.
    """
    identities: frozenset[str]
    sign: int
    lower: float
    upper: float

    def __post_init__(self):
        if self.sign not in (-1, 1):
            raise ValueError("a delta slice has insertion or retraction sign")
        if not (isfinite(self.lower) and isfinite(self.upper)):
            raise ValueError("slice endpoints must be finite")
        if self.lower < 0 or self.lower > self.upper:
            raise ValueError("invalid nonnegative slice interval")


@dataclass(frozen=True)
class Anchor:
    """Immutable interval for one frozen cell population."""
    identity: str
    population: tuple[Occurrence, ...]
    lower: float
    upper: float
    group: str = "group"

    def __post_init__(self):
        keys = [item.identity for item in self.population]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate occurrence in an anchor frame")
        if any(item.group != self.group for item in self.population):
            raise ValueError("an anchor frame must belong to one group")
        mass = sum(item.weight for item in self.population)
        if not (isfinite(self.lower) and isfinite(self.upper)):
            raise ValueError("anchor endpoints must be finite")
        if not 0 <= self.lower <= self.upper <= mass + 1e-9:
            raise ValueError("anchor interval lies outside its physical mass")

    def project(self, residual: Mapping[str, Occurrence], ledger: LabelLedger,
                slices: Iterable[SignedSlice] = ()):
        """Replay net signed population differences from this anchor.

        Replaying from the immutable anchor is essential. Reusing the clipped
        result of an earlier projection would discard information when an old
        unknown deletion is subsequently resolved.
        """
        frame = {item.identity: item for item in self.population}
        current = dict(residual)
        if any(item.group != self.group for item in current.values()):
            raise ValueError("residual contains another output group")
        for key in frame.keys() & current.keys():
            if frame[key] != current[key]:
                raise ValueError("an occurrence identity changed across versions")
        removed = {key: frame[key] for key in frame.keys() - current.keys()}
        added = {key: current[key] for key in current.keys() - frame.keys()}
        lower, upper = self.lower, self.upper
        consumed: set[tuple[int, str]] = set()

        # Replace disjoint raw leaves with a certified total for that slice.
        for certificate in slices:
            population = added if certificate.sign == 1 else removed
            if not certificate.identities.issubset(population):
                raise IncompatibleSliceFrame("frozen slice is no longer a subset of its net delta side")
            if any((certificate.sign, key) in consumed for key in certificate.identities):
                raise ValueError("overlapping delta certificates would double-count")
            members = [population[key] for key in certificate.identities]
            mass = sum(item.weight for item in members)
            if certificate.upper > mass + 1e-9:
                raise ValueError("slice interval exceeds its physical mass")
            known = sum(item.weight * ledger.labels[item.semantic_key]
                        for item in members if item.semantic_key in ledger.labels)
            unknown_mass = sum(item.weight for item in members
                               if item.semantic_key not in ledger.labels)
            slice_lower = max(certificate.lower, known)
            slice_upper = min(certificate.upper, known + unknown_mass)
            if slice_lower > slice_upper + 1e-9:
                raise ValueError("slice certificate conflicts with exact labels")
            if certificate.sign == 1:
                lower += slice_lower
                upper += slice_upper
            else:
                lower -= slice_upper
                upper -= slice_lower
            consumed.update((certificate.sign, key) for key in certificate.identities)

        for sign, population in ((-1, removed), (1, added)):
            for key, item in population.items():
                if (sign, key) in consumed:
                    continue
                if item.semantic_key in ledger.labels:
                    contribution = item.weight * ledger.labels[item.semantic_key]
                    lower += sign * contribution
                    upper += sign * contribution
                elif sign == -1:
                    lower -= item.weight
                else:
                    upper += item.weight

        mass = sum(item.weight for item in current.values())
        lower, upper = max(0.0, lower), min(mass, upper)
        if lower > upper + 1e-9:
            raise ValueError("empty projection: certificate failure or corrupt lineage")
        return lower, upper


def composed_interval(snapshot: Snapshot, ledger: LabelLedger,
                      anchors: Iterable[Anchor], group="group"):
    """Intersect anchor projections, then add current exact contribution."""
    residual = snapshot.residual(ledger, group)
    mass = sum(item.weight for item in residual.values())
    lower, upper = 0.0, mass
    for anchor in anchors:
        if anchor.group != group:
            raise ValueError("incompatible anchor group")
        lo, hi = anchor.project(residual, ledger)
        lower, upper = max(lower, lo), min(upper, hi)
    if lower > upper + 1e-9:
        raise ValueError("incompatible projected anchors")
    exact = snapshot.exact_total(ledger, group)
    return exact + lower, exact + upper
