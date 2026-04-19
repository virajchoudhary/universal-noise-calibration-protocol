"""Table + figure generation for UNCP results."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def results_table(rows: List[Dict], columns: List[str]) -> pd.DataFrame:
    return pd.DataFrame(rows)[columns]


def to_latex(df: pd.DataFrame, caption: str, label: str,
             bold_best: bool = True) -> str:
    """Format dataframe as booktabs LaTeX table with best value bolded."""
    out = df.copy()
    if bold_best:
        numeric = out.select_dtypes("number")
        if not numeric.empty:
            best_col = numeric.columns[-1]
            best = numeric[best_col].max()
            out[best_col] = [f"\\textbf{{{v:.3f}}}" if v == best else f"{v:.3f}"
                             for v in out[best_col]]
    body = out.to_latex(index=False, escape=False, column_format="l" + "c" * (len(out.columns) - 1))
    return (
        f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n{body}\\end{{table}}\n"
    )
