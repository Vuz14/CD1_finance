from __future__ import annotations

import logging
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger("cd1_finance.preprocessing")


def _norm(text: object) -> str:
    value = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return value.lower().strip()


class FinancialIndicatorCalculator:
    """Tính X1-X13 từ báo cáo tài chính, giữ nguyên định nghĩa toán học hiện tại."""

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for i in range(1, 14):
            out[f"X{i}"] = 0.0
        cols = FinancialIndicatorCalculator._find_columns(out)
        for idx in out.index:
            revenue = FinancialIndicatorCalculator._get_value(out, idx, cols["revenue"], 1.0)
            cogs = FinancialIndicatorCalculator._get_value(out, idx, cols["cogs"], 0.0)
            gross_profit = FinancialIndicatorCalculator._get_value(out, idx, cols["gross_profit"], revenue - cogs)
            pretax_income = FinancialIndicatorCalculator._get_value(out, idx, cols["pretax_income"], 0.0)
            total_assets = FinancialIndicatorCalculator._get_value(out, idx, cols["total_assets"], 1.0)
            equity = FinancialIndicatorCalculator._get_value(out, idx, cols["equity"], 1.0)
            total_liabilities = FinancialIndicatorCalculator._get_value(out, idx, cols["total_liabilities"], max(0.0, total_assets - equity))
            current_assets = FinancialIndicatorCalculator._get_value(out, idx, cols["current_assets"], 0.0)
            current_liabilities = FinancialIndicatorCalculator._get_value(out, idx, cols["current_liabilities"], 1.0)
            inventory = FinancialIndicatorCalculator._get_value(out, idx, cols["inventory"], 0.0)
            cash = FinancialIndicatorCalculator._get_value(out, idx, cols["cash"], 0.0)
            receivables = FinancialIndicatorCalculator._get_value(out, idx, cols["receivables"], 0.0)
            interest_expense = FinancialIndicatorCalculator._get_value(out, idx, cols["interest_expense"], 0.0)
            depreciation = FinancialIndicatorCalculator._get_value(out, idx, cols["depreciation"], 0.0)

            out.at[idx, "X1"] = gross_profit / revenue if revenue else 0.0
            out.at[idx, "X2"] = pretax_income / revenue if revenue else 0.0
            out.at[idx, "X3"] = pretax_income / total_assets if total_assets else 0.0
            out.at[idx, "X4"] = pretax_income / equity if equity else 0.0
            out.at[idx, "X5"] = total_liabilities / total_assets if total_assets else 0.0
            out.at[idx, "X6"] = current_assets / current_liabilities if current_liabilities else 0.0
            out.at[idx, "X7"] = (current_assets - inventory) / current_liabilities if current_liabilities else 0.0
            out.at[idx, "X8"] = (pretax_income + interest_expense) / interest_expense if interest_expense else 0.0
            out.at[idx, "X9"] = (pretax_income + interest_expense + depreciation) / total_liabilities if total_liabilities else 0.0
            out.at[idx, "X10"] = cash / current_liabilities if current_liabilities else 0.0
            out.at[idx, "X11"] = cogs / inventory if inventory else 0.0
            out.at[idx, "X12"] = revenue / receivables if receivables else 0.0
            out.at[idx, "X13"] = revenue / total_assets if total_assets else 0.0
        return out

    @staticmethod
    def _find_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
        patterns = {
            "revenue": ["doanh thu thuan", "tong doanh thu", "thu nhap lai thuan"],
            "cogs": ["gia von hang ban", "chi phi lai"],
            "gross_profit": ["loi nhuan tu hdkd", "loi nhuan gop"],
            "pretax_income": ["loi nhuan truoc thue"],
            "total_assets": ["tong cong tai san", "tong tai san"],
            "equity": ["von chu so huu"],
            "total_liabilities": ["tong no phai tra"],
            "current_assets": ["tai san ngan han", "tai san luu dong", "tien gui khach hang"],
            "current_liabilities": ["no phai tra ngan han", "cac khoan no"],
            "inventory": ["hang ton kho"],
            "cash": ["tien mat", "tien va cac khoan tuong duong tien"],
            "receivables": ["phai thu"],
            "interest_expense": ["chi phi lai vay", "chi phi lai"],
            "depreciation": ["hao mon", "khau hao"],
        }
        found: Dict[str, Optional[str]] = {k: None for k in patterns}
        normalized = {col: _norm(col) for col in df.columns}
        for key, keys in patterns.items():
            for col, text in normalized.items():
                if any(p in text for p in keys):
                    found[key] = col
                    break
        return found

    @staticmethod
    def _get_value(df: pd.DataFrame, idx: int, col_name: Optional[str], default: float = 0.0) -> float:
        if not col_name or col_name not in df.columns:
            return float(default)
        value = pd.to_numeric(df.at[idx, col_name], errors="coerce")
        return float(default if pd.isna(value) else value)


class DataPreprocessor:
    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        x_cols = [f"X{i}" for i in range(1, 14) if f"X{i}" in out.columns]
        if not x_cols:
            return out
        out[x_cols] = out[x_cols].apply(pd.to_numeric, errors="coerce")
        if out[x_cols].isna().sum().sum() > 0:
            out[x_cols] = KNNImputer(n_neighbors=min(5, max(1, len(out) - 1)), weights="distance").fit_transform(out[x_cols])
        for col in x_cols:
            q1, q3 = out[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            out[col] = out[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            out[col] = out[col].clip(out[col].quantile(0.05), out[col].quantile(0.95))
        out[x_cols] = StandardScaler().fit_transform(out[x_cols])
        return out


def process_raw_data(file_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(file_path) if file_path.lower().endswith(".csv") else pd.read_excel(file_path)
    rename_map = {}
    for col in df.columns:
        text = _norm(col)
        if text in {"ma cp", "macp", "ma chung khoan"}:
            rename_map[col] = "MaCP"
        if text in {"nam", "year"}:
            rename_map[col] = "Year"
    if rename_map:
        df = df.rename(columns=rename_map)
    if "STT" not in df.columns:
        df.insert(0, "STT", range(1, len(df) + 1))
    out = FinancialIndicatorCalculator.calculate_indicators(df)
    out = DataPreprocessor.preprocess(out)
    keep = [c for c in ["STT", "MaCP", "Year"] + [f"X{i}" for i in range(1, 14)] if c in out.columns]
    out = out[keep]
    if output_path:
        out.to_csv(output_path, index=False, encoding="utf-8-sig")
    return out
