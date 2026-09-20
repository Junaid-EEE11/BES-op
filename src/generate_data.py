"""Compatibility wrapper for the synthetic-data generator."""

from pathlib import Path

from bessopt.data import generate_synthetic_data


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "synthetic_bess_data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data(days=30)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
