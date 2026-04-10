from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
import random

from experiment.schemas.tier1 import (
    MEMBER_SPLIT,
    NON_MEMBER_SPLIT,
    CanaryRegistryEntry,
    Tier1Record,
)

from .config import Tier1DataConfig


FIRST_NAMES = (
    "Aaliyah",
    "Adrian",
    "Alex",
    "Alyssa",
    "Amir",
    "Ava",
    "Benjamin",
    "Brianna",
    "Caleb",
    "Camila",
    "Charlotte",
    "Chloe",
    "Daniel",
    "Darius",
    "David",
    "Elena",
    "Elijah",
    "Ella",
    "Emily",
    "Ethan",
    "Gabriel",
    "Grace",
    "Hannah",
    "Harper",
    "Henry",
    "Isabella",
    "Jasmine",
    "Jayden",
    "Joseph",
    "Joshua",
    "Layla",
    "Leah",
    "Liam",
    "Lucas",
    "Mason",
    "Mia",
    "Mila",
    "Naomi",
    "Natalie",
    "Noah",
    "Nora",
    "Olivia",
    "Owen",
    "Riley",
    "Samuel",
    "Scarlett",
    "Sebastian",
    "Sofia",
    "Sophia",
    "Wyatt",
)

LAST_NAMES = (
    "Adams",
    "Allen",
    "Anderson",
    "Baker",
    "Brooks",
    "Brown",
    "Campbell",
    "Carter",
    "Castillo",
    "Chang",
    "Coleman",
    "Collins",
    "Davis",
    "Diaz",
    "Evans",
    "Flores",
    "Garcia",
    "Gonzalez",
    "Green",
    "Hall",
    "Harris",
    "Hernandez",
    "Hill",
    "Jackson",
    "Johnson",
    "Kim",
    "Lee",
    "Lewis",
    "Martinez",
    "Miller",
    "Mitchell",
    "Moore",
    "Nguyen",
    "Parker",
    "Patel",
    "Perez",
    "Ramirez",
    "Rivera",
    "Roberts",
    "Robinson",
    "Sanchez",
    "Scott",
    "Smith",
    "Taylor",
    "Thomas",
    "Thompson",
    "Turner",
    "Walker",
    "Williams",
    "Young",
)

MIDDLE_INITIALS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NAME_SUFFIXES = ("Jr.", "Sr.", "III")

MERCHANT_CATEGORIES = (
    "grocery",
    "fuel",
    "restaurant",
    "travel",
    "electronics",
    "healthcare",
    "home_improvement",
    "online_marketplace",
    "utilities",
    "wire_transfer",
)

CATEGORY_AMOUNT_RANGES = {
    "grocery": (18.0, 320.0, 85.0),
    "fuel": (22.0, 180.0, 68.0),
    "restaurant": (15.0, 260.0, 72.0),
    "travel": (140.0, 4200.0, 980.0),
    "electronics": (45.0, 3600.0, 540.0),
    "healthcare": (25.0, 1600.0, 210.0),
    "home_improvement": (30.0, 2400.0, 420.0),
    "online_marketplace": (12.0, 1500.0, 220.0),
    "utilities": (40.0, 540.0, 160.0),
    "wire_transfer": (180.0, 4800.0, 1300.0),
}

GEO_LOCATIONS = (
    "Atlanta, GA, USA",
    "Austin, TX, USA",
    "Baltimore, MD, USA",
    "Boston, MA, USA",
    "Charlotte, NC, USA",
    "Chicago, IL, USA",
    "Dallas, TX, USA",
    "Denver, CO, USA",
    "Detroit, MI, USA",
    "Houston, TX, USA",
    "Las Vegas, NV, USA",
    "Los Angeles, CA, USA",
    "Miami, FL, USA",
    "Minneapolis, MN, USA",
    "Nashville, TN, USA",
    "New York, NY, USA",
    "Orlando, FL, USA",
    "Phoenix, AZ, USA",
    "Portland, OR, USA",
    "San Diego, CA, USA",
    "San Francisco, CA, USA",
    "Seattle, WA, USA",
    "St. Louis, MO, USA",
    "Washington, DC, USA",
)

DEVICE_CHANNELS = ("ios", "android", "web", "desktop")
HIGH_RISK_CATEGORIES = {"electronics", "travel", "online_marketplace", "wire_transfer"}
HIGH_RISK_GEOS = {"Las Vegas, NV, USA", "Miami, FL, USA", "Orlando, FL, USA"}


@dataclass(frozen=True, slots=True)
class GeneratedTier1Dataset:
    records: tuple[Tier1Record, ...]
    canary_registry: tuple[CanaryRegistryEntry, ...]


def generate_tier1_dataset(config: Tier1DataConfig) -> GeneratedTier1Dataset:
    member_indices = _select_member_indices(config)
    canary_ids_by_index = _assign_canary_ids(config, member_indices)

    drafts: list[dict[str, object]] = []
    risk_scores: list[tuple[int, float]] = []
    for index in range(config.total_records):
        split = MEMBER_SPLIT if index in member_indices else NON_MEMBER_SPLIT
        canary_id = canary_ids_by_index.get(index)
        draft = _build_record_draft(config, index=index, split=split, canary_id=canary_id)
        drafts.append(draft)
        risk_scores.append((index, _compute_risk_score(draft, config=config, index=index)))

    fraud_indices = _select_fraud_indices(config, risk_scores)

    records: list[Tier1Record] = []
    canary_registry: list[CanaryRegistryEntry] = []
    for index, draft in enumerate(drafts):
        record = Tier1Record(
            record_id=str(draft["record_id"]),
            account_id=str(draft["account_id"]),
            customer_name=str(draft["customer_name"]),
            date_of_birth=str(draft["date_of_birth"]),
            account_number=str(draft["account_number"]),
            amount=float(draft["amount"]),
            merchant_category=str(draft["merchant_category"]),
            timestamp=str(draft["timestamp"]),
            geo_location=str(draft["geo_location"]),
            device_fingerprint=str(draft["device_fingerprint"]),
            is_fraud_label=1 if index in fraud_indices else 0,
            split=str(draft["split"]),
            is_canary=bool(draft["is_canary"]),
            canary_id=str(draft["canary_id"]) if draft["canary_id"] is not None else None,
        )
        records.append(record)
        if record.is_canary:
            canary_registry.append(
                CanaryRegistryEntry(
                    canary_id=record.canary_id or "",
                    record_id=record.record_id,
                    customer_name=record.customer_name,
                    date_of_birth=record.date_of_birth,
                    account_number=record.account_number,
                )
            )

    return GeneratedTier1Dataset(records=tuple(records), canary_registry=tuple(canary_registry))


def _select_member_indices(config: Tier1DataConfig) -> set[int]:
    population = list(range(config.total_records))
    rng = _rng_for(config.seed, "member_split")
    rng.shuffle(population)
    return set(population[: config.member_records])


def _assign_canary_ids(config: Tier1DataConfig, member_indices: set[int]) -> dict[int, str]:
    ordered_members = sorted(member_indices)
    rng = _rng_for(config.seed, "canary_selection")
    selected_indices = sorted(rng.sample(ordered_members, config.canary_count))
    return {index: f"CANARY-{offset:03d}" for offset, index in enumerate(selected_indices, start=1)}


def _build_record_draft(
    config: Tier1DataConfig,
    *,
    index: int,
    split: str,
    canary_id: str | None,
) -> dict[str, object]:
    merchant_category = _pick_from_vocab(config, index, "merchant", MERCHANT_CATEGORIES)
    geo_location = _pick_from_vocab(config, index, "geo", GEO_LOCATIONS)
    customer_name = _generate_customer_name(config, index)
    date_of_birth = _generate_date(config, index, field_name="dob", start=config.dob_start, end=config.dob_end)
    timestamp = _generate_timestamp(config, index)

    return {
        "record_id": f"REC-{index + 1:06d}",
        "account_id": f"ACC-{10000 + index:05d}",
        "customer_name": customer_name,
        "date_of_birth": date_of_birth.isoformat(),
        "account_number": _generate_account_number(index),
        "amount": _generate_amount(config, index, merchant_category),
        "merchant_category": merchant_category,
        "timestamp": timestamp,
        "geo_location": geo_location,
        "device_fingerprint": _generate_device_fingerprint(config, index),
        "split": split,
        "is_canary": canary_id is not None,
        "canary_id": canary_id,
    }


def _generate_customer_name(config: Tier1DataConfig, index: int) -> str:
    rng = _rng_for(config.seed, index, "customer_name")
    first_name = FIRST_NAMES[rng.randrange(len(FIRST_NAMES))]
    last_name = LAST_NAMES[rng.randrange(len(LAST_NAMES))]

    parts = [first_name]
    if rng.random() < 0.38:
        parts.append(f"{MIDDLE_INITIALS[rng.randrange(len(MIDDLE_INITIALS))]}.")
    parts.append(last_name)
    if rng.random() < 0.06:
        parts.append(NAME_SUFFIXES[rng.randrange(len(NAME_SUFFIXES))])
    return " ".join(parts)


def _generate_date(
    config: Tier1DataConfig,
    index: int,
    *,
    field_name: str,
    start: date,
    end: date,
) -> date:
    days = (end - start).days
    rng = _rng_for(config.seed, index, field_name)
    return start + timedelta(days=rng.randint(0, days))


def _generate_timestamp(config: Tier1DataConfig, index: int) -> str:
    total_seconds = int((config.timestamp_end - config.timestamp_start).total_seconds())
    rng = _rng_for(config.seed, index, "timestamp")
    offset_seconds = rng.randint(0, total_seconds)
    timestamp = config.timestamp_start + timedelta(seconds=offset_seconds)
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _generate_account_number(index: int) -> str:
    base_number = 1_000_000_000 + index
    checksum = 97 - (base_number % 97)
    return f"{checksum:02d}{base_number:010d}"


def _generate_amount(config: Tier1DataConfig, index: int, merchant_category: str) -> float:
    minimum, maximum, mode = CATEGORY_AMOUNT_RANGES[merchant_category]
    rng = _rng_for(config.seed, index, "amount")
    return round(rng.triangular(minimum, maximum, mode), 2)


def _generate_device_fingerprint(config: Tier1DataConfig, index: int) -> str:
    rng = _rng_for(config.seed, index, "device")
    channel = DEVICE_CHANNELS[rng.randrange(len(DEVICE_CHANNELS))]
    digest = sha256(f"{config.seed}|device|{index}".encode("utf-8")).hexdigest()
    return f"dfp-{channel}-{digest[:8]}-{digest[8:16]}-{digest[16:24]}"


def _pick_from_vocab(config: Tier1DataConfig, index: int, label: str, vocabulary: tuple[str, ...]) -> str:
    rng = _rng_for(config.seed, index, label)
    return vocabulary[rng.randrange(len(vocabulary))]


def _compute_risk_score(draft: dict[str, object], *, config: Tier1DataConfig, index: int) -> float:
    amount = float(draft["amount"])
    merchant_category = str(draft["merchant_category"])
    geo_location = str(draft["geo_location"])
    hour = int(str(draft["timestamp"])[11:13])

    score = 0.0
    score += min(amount / 2200.0, 1.8)
    if merchant_category in HIGH_RISK_CATEGORIES:
        score += 1.1
    if geo_location in HIGH_RISK_GEOS:
        score += 0.7
    if hour < 5 or hour >= 23:
        score += 0.85
    elif hour < 7 or hour >= 21:
        score += 0.35

    jitter = _rng_for(config.seed, index, "fraud_score_jitter").random()
    return score + (jitter * 0.4)


def _select_fraud_indices(config: Tier1DataConfig, risk_scores: list[tuple[int, float]]) -> set[int]:
    fraud_count = round(config.total_records * config.fraud_base_rate)
    ordered = sorted(risk_scores, key=lambda item: (-item[1], item[0]))
    return {index for index, _ in ordered[:fraud_count]}


def _rng_for(seed: int, *parts: object) -> random.Random:
    material = "|".join([str(seed), *(str(part) for part in parts)])
    digest = sha256(material.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], byteorder="big", signed=False))
