"""Monte Carlo stock price simulation based on historical CSV data.

This script reads the CSV structure described in README.md and simulates
future price paths for a chosen ticker using a geometric Brownian motion
model calibrated from historical log returns.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class PriceSeries:
    ticker: str
    dates: Sequence[str]
    closes: Sequence[float]

    def log_returns(self) -> List[float]:
        if len(self.closes) < 2:
            raise ValueError("At least two closing prices are required for returns")
        return [math.log(self.closes[i] / self.closes[i - 1]) for i in range(1, len(self.closes))]


def parse_columns(rows: Sequence[Sequence[str]]) -> Dict[Tuple[str, str], int]:
    if len(rows) < 2:
        raise ValueError("CSV does not contain the expected header rows")
    metrics = rows[0][1:]
    tickers = rows[1][1:]
    if len(metrics) != len(tickers):
        raise ValueError("Header rows have mismatched lengths")
    return {(ticker, metric): idx + 1 for idx, (metric, ticker) in enumerate(zip(metrics, tickers))}


def load_price_series(csv_path: Path, ticker: str) -> PriceSeries:
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]

    column_map = parse_columns(rows)
    key = (ticker, "Close")
    if key not in column_map:
        available = sorted({t for t, _ in column_map})
        raise ValueError(f"Ticker '{ticker}' not found. Available tickers: {', '.join(available)}")

    date_start = 3  # first three rows are headers per README description
    date_rows = [row for row in rows[date_start:] if row and row[0]]

    dates: List[str] = []
    closes: List[float] = []
    close_idx = column_map[key]

    for row in date_rows:
        if len(row) <= close_idx or row[close_idx] == "":
            continue
        try:
            close_value = float(row[close_idx])
        except ValueError:
            continue
        dates.append(row[0])
        closes.append(close_value)

    if not closes:
        raise ValueError(f"No close prices found for ticker {ticker}")

    return PriceSeries(ticker=ticker, dates=dates, closes=closes)


def simulate_paths(
    start_price: float,
    drift: float,
    volatility: float,
    days: int,
    paths: int,
    seed: int | None = None,
) -> List[List[float]]:
    if seed is not None:
        random.seed(seed)

    if volatility < 0:
        raise ValueError("Volatility cannot be negative")
    if days <= 0 or paths <= 0:
        raise ValueError("'days' and 'paths' must be positive integers")

    dt = 1.0  # daily steps with daily historical statistics
    paths_data: List[List[float]] = []
    exp_drift = drift - 0.5 * volatility * volatility

    for _ in range(paths):
        price = start_price
        path = [price]
        for _ in range(days):
            shock = random.gauss(0.0, 1.0)
            price *= math.exp(exp_drift * dt + volatility * math.sqrt(dt) * shock)
            path.append(price)
        paths_data.append(path)

    return paths_data


def summarize_paths(paths: Sequence[Sequence[float]]) -> Dict[str, float]:
    final_prices = [path[-1] for path in paths]
    sorted_final = sorted(final_prices)
    count = len(sorted_final)

    def percentile(p: float) -> float:
        if count == 1:
            return sorted_final[0]
        idx = p * (count - 1)
        lower = math.floor(idx)
        upper = math.ceil(idx)
        if lower == upper:
            return sorted_final[int(idx)]
        frac = idx - lower
        return sorted_final[lower] * (1 - frac) + sorted_final[upper] * frac

    return {
        "mean": fmean(final_prices),
        "min": min(final_prices),
        "max": max(final_prices),
        "p05": percentile(0.05),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
    }


def format_currency(value: float) -> str:
    return f"{value:,.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo stock price simulation")
    parser.add_argument("--csv", type=Path, default=Path("temp.csv"), help="Path to the historical CSV file")
    parser.add_argument("--ticker", required=True, help="Ticker symbol to simulate (e.g., AAPL)")
    parser.add_argument("--days", type=int, default=252, help="Number of future trading days to simulate")
    parser.add_argument("--paths", type=int, default=1000, help="Number of Monte Carlo paths")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible runs")
    args = parser.parse_args()

    series = load_price_series(args.csv, args.ticker)
    log_returns = series.log_returns()
    drift = fmean(log_returns)
    volatility = pstdev(log_returns)

    paths = simulate_paths(series.closes[-1], drift, volatility, args.days, args.paths, args.seed)
    summary = summarize_paths(paths)

    print(f"Historical points loaded: {len(series.closes)}")
    print(f"Latest close: {format_currency(series.closes[-1])}")
    print(f"Drift (daily mean log return): {drift:.6f}")
    print(f"Volatility (daily log-return σ): {volatility:.6f}")
    print()
    print(f"Simulated {args.paths} paths for {args.days} days.")
    print("Final price distribution (currency units):")
    for label in ["min", "p05", "p50", "mean", "p95", "max"]:
        print(f"  {label:>4}: {format_currency(summary[label])}")
    print()
    sample_path = paths[0]
    print("Sample path (first 5 days):")
    for day, price in enumerate(sample_path[:6]):
        label = "Start" if day == 0 else f"Day {day}"
        print(f"  {label:>5}: {format_currency(price)}")


if __name__ == "__main__":
    main()
