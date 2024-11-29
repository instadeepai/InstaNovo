from __future__ import annotations

import argparse
import logging

from instanovo.utils import SpectrumDataFrame

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main() -> None:
    """Convert data to spectrum data frame and save as parquet."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument("source", help="source file(s)")
    parser.add_argument("target", help="target folder to save data shards")
    parser.add_argument(
        "--is_annotated",
        default=False,
        help="whether dataset is annotated",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="name of saved dataset",
    )
    parser.add_argument(
        "--partition",
        default=None,
        choices=["train", "valid", "test"],
        help="partition of saved dataset",
    )
    parser.add_argument(
        "--shard_size",
        default=1_000_000,
        help="length of saved data shards",
    )
    parser.add_argument("--max_charge", default=10, help="maximum charge to filter out")

    args = parser.parse_args()

    logger.info(f"Loading {args.source}")
    sdf = SpectrumDataFrame.load(
        args.source,
        is_annotated=bool(args.is_annotated),
        name=args.name,
        partition=args.partition,
        max_shard_size=int(args.shard_size),
        lazy=True,
    )
    logger.info(f"Loaded {len(sdf):,d} rows")

    logger.info(f"Filtering max_charge <= {int(args.max_charge)}")
    sdf.filter_rows(lambda row: row["precursor_charge"] <= int(args.max_charge))

    logger.info(f"Saving {len(sdf):,d} rows to {args.target}")
    sdf.save(
        args.target,
        name=args.name,
        partition=args.partition,
        max_shard_size=int(args.shard_size),
    )

    logger.info("Saving complete.")
    del sdf


if __name__ == "__main__":
    main()
