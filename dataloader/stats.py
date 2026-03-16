from typing import Dict, List


def print_split_stats(split: str, slices: Dict):
    """Print a concise stats table for all language slices in a split."""
    print(f"\n  [{split}] language statistics")
    print(
        f"  {'Language':<12} {'Rows':>7}  {'Hours':>7}  "
        f"{'AvgDur':>7}  {'AvgSNR':>7}  {'Speakers':>9}"
    )
    print(f"  {'-'*12} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}")

    for lang, ds in slices.items():
        n = len(ds)
        durs = ds["duration"]
        snrs = ds["snr_db"]
        users = set(ds["user_id"])
        hours = sum(durs) / 3600
        avg_d = sum(durs) / n
        avg_s = sum(snrs) / n
        print(
            f"  {lang:<12} {n:>7,}  {hours:>7.1f}  "
            f"{avg_d:>7.2f}  {avg_s:>7.2f}  {len(users):>9}"
        )


def print_pair_coverage(datasets: list):
    """
    Print pairing coverage for a list of S2TTDataset instances.
    Shows how many source rows have a confirmed English translation.
    """
    print(f"\n  S2TT pair coverage")
    print(
        f"  {'Language':<12} {'Paired':>8}  {'Skipped':>8}  "
        f"{'%Paired':>8}  {'Hours':>7}  {'Sources'}"
    )
    print(f"  {'-'*12} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*20}")

    for ds in datasets:
        s = ds.summary()
        pct = 100 * s["paired_rows"] / max(1, s["paired_rows"] + s["skipped_rows"])
        print(
            f"  {s['source_language']:<12} "
            f"{s['paired_rows']:>8,}  "
            f"{s['skipped_rows']:>8,}  "
            f"{pct:>8.1f}  "
            f"{s['total_hours']:>7.1f}  "
            f"{s['source_codes']}"
        )
