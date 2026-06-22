from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from docx import Document


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "backend" / "results"
SOURCE = Path(r"D:\AI\Kì 5\Chuyên đề 1\Report.docx")
OUTPUT = ROOT / "docs" / "reports" / "Report_revised.docx"


def set_paragraph(document: Document, index: int, text: str) -> None:
    document.paragraphs[index].text = text


def remove_paragraph(paragraph) -> None:
    element = paragraph._element
    element.getparent().remove(element)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def metrics(model: str) -> Dict[str, object]:
    path = RESULTS / "research_holdout" / model / "metrics_holdout.json"
    return json.loads(path.read_text(encoding="utf-8"))


def ranking(model: str) -> List[str]:
    path = RESULTS / "research_holdout" / model / "model_specific_permutation_importance.csv"
    return pd.read_csv(path)["feature"].tolist()


def main() -> None:
    document = Document(SOURCE)
    title = "MACHINE LEARNING-BASED PREDICTION OF FINANCIAL DISTRESS FOR VIETNAMESE LISTED COMPANIES, 2016–2023: A TEMPORAL EVALUATION"
    set_paragraph(document, 7, title)
    set_paragraph(document, 23, title)
    set_paragraph(
        document,
        38,
        "This study investigates early prediction of financial distress among Vietnamese listed companies using firm-year financial ratios from 2016–2023. Financial distress is treated as an observable regulatory and accounting proxy rather than legal bankruptcy or debt default.",
    )
    set_paragraph(
        document,
        41,
        "The empirical objective is to compare machine-learning models under temporal evaluation. Financial information at year T is used to predict a distress status observed at T+1; observations for feature year 2024 are excluded because the corresponding 2025 outcome is incomplete.",
    )
    set_paragraph(
        document,
        42,
        "The study compares five implemented approaches. Calibrated multivariate Logistic Regression provides an interpretable linear baseline; ANN, LightGBM, and XGBoost represent nonlinear alternatives; and the temporal stacking ensemble combines their predictions without using future-year labels.",
    )
    set_paragraph(
        document,
        43,
        "The implemented comparison includes Artificial Neural Network, calibrated multivariate Logistic Regression, LightGBM, XGBoost, and a temporal stacking ensemble. LightGBM and XGBoost represent nonlinear boosting methods, while Logistic Regression is the interpretable statistical baseline.",
    )
    set_paragraph(
        document,
        47,
        "The reconstructed research cohort contains 1,162 firm-year observations from 152 Vietnamese listed companies during 2016–2023, with 21 positive T+1 distress labels. Development data contain 882 observations through 2021; the retrospective 2022–2023 holdout contains 280 observations, including 12 positives.",
    )
    set_paragraph(
        document,
        49,
        "The target is a T+1 financial-distress proxy. A firm is labeled distressed at event year T+1 when source evidence documents at least one of the following: losses for three consecutive years, accumulated losses exceeding actually contributed charter capital, or negative shareholders’ equity. Financial ratios observed at year T are used as predictors. This definition must not be interpreted as legal bankruptcy or a verified debt default. Each positive label should be traceable to its triggering criterion and source document.",
    )
    set_paragraph(
        document,
        50,
        "The eight ratios and T+1 labels are reconstructed from BCTC_2015_2024_TongHop.csv. Among 48 duplicate company-year groups, label inputs agree; the feature record with the smallest accounting-identity error is retained. Total liabilities are reconstructed as total assets minus equity because the scraped liabilities field is shifted for many rows. Rows with missing ratio inputs are excluded. No global scaling, imputation, or winsorization is performed before temporal splitting.",
    )
    set_paragraph(
        document,
        52,
        "Random stratified splitting is not used. Development data cover 2016–2021 and 2022–2023 form a retrospective temporal holdout. The feature year 2015 is excluded because 2014 net income is unavailable for evaluating the three-consecutive-loss criterion at event year 2016. Feature year 2024 is excluded because event-year 2025 statements are unavailable. Outlier clipping and scaling are fitted inside each training fold.",
    )
    set_paragraph(document, 54, "Table 1: Predictor Variables in the Financial-Distress Model")
    set_paragraph(document, 60, "LightGBM classification objective")
    set_paragraph(document, 61, "Weighted Binary Log-Loss")
    set_paragraph(
        document,
        63,
        "For binary financial-distress classification, the objective follows maximum-likelihood estimation for a Bernoulli outcome.",
    )
    set_paragraph(
        document,
        71,
        "The boosting classifier maps its raw score to a probability through the sigmoid link function.",
    )
    set_paragraph(document, 102, "Random Forest background (not an evaluated holdout model)")
    set_paragraph(
        document,
        134,
        "The hybrid feature-selection score combines normalized ExtraTrees importance with a normalized GRA score; it is distinct from model-specific interpretation.",
    )
    set_paragraph(
        document,
        225,
        "Hyperparameters are explicitly recorded in backend/config/training_params.json for reproducibility. They are not presented as Optuna-optimal because the current repository does not contain a reproducible nested temporal Optuna study. Model selection claims are therefore limited to the implemented temporal experiments and ablation analysis.",
    )
    set_paragraph(document, 226, "Table 2: Final recorded hyperparameter settings used in the experiments.")
    set_paragraph(
        document,
        229,
        "Logistic Regression uses L2 regularization and sigmoid calibration with three-fold cross-validation. This configuration is an explicitly recorded experimental setting, not evidence that an elastic-net optimum was found.",
    )
    set_paragraph(
        document,
        231,
        "These settings provide a reproducible baseline for the eight-ratio, highly imbalanced panel. Their superiority is not claimed without nested temporal tuning and validation on a new cohort.",
    )
    set_paragraph(
        document,
        233,
        "Primary performance is measured on the 2022–2023 retrospective temporal holdout. AUC, PR-AUC, F1, F2, G-Mean, Brier score, expected calibration error, confusion matrices, and bootstrap confidence intervals are reported. Decision thresholds are selected from expanding-window development predictions through 2021.",
    )
    set_paragraph(
        document,
        236,
        "LightGBM and XGBoost provide the strongest discrimination on the temporal holdout, both with AUC 0.9717. Their AUC difference is not statistically significant under a paired bootstrap (difference 0.0000; 95% CI [−0.0157, 0.0185]).",
    )
    set_paragraph(
        document,
        237,
        "Because only 12 of 280 holdout observations are positive, PR-AUC and recall are emphasized alongside AUC. XGBoost achieves PR-AUC 0.6457 and detects 8 of 12 distress cases at its development-selected F2 threshold; LightGBM achieves PR-AUC 0.5994 and detects 7 of 12.",
    )
    set_paragraph(
        document,
        238,
        "Ablation analysis supports the primary configuration using all eight ratios without SMOTE-ENN. Feature-selected and resampled variants perform worse on the retrospective holdout. Because this comparison was made after viewing the holdout, it is reported as exploratory and requires confirmation on a new external cohort.",
    )
    set_paragraph(
        document,
        239,
        "Thresholds are estimated only from past development predictions. They are operating-point choices reflecting the asymmetric cost of false negatives and must be reported together with confusion matrices; they are not universal probabilities of distress.",
    )
    set_paragraph(
        document,
        240,
        "The results support the feasibility of an early-warning model for Vietnamese listed companies but do not establish production readiness. Remaining requirements include row-level label verification, external validation, sector analysis, and reproducible hyperparameter optimization.",
    )
    set_paragraph(
        document,
        242,
        "Model-specific interpretation is performed using permutation importance on the temporal holdout with average precision as the scoring metric. The separate GRA+ExtraTrees hybrid score is used for feature selection and must not be described as model-specific feature importance.",
    )
    set_paragraph(
        document,
        243,
        "Permutation importance is recalculated separately for each fitted holdout model using average precision. The hybrid GRA+ExtraTrees ranking shown in the final column is retained only to document the feature-selection pipeline; it should not be interpreted as a causal effect or as the native importance of ANN, Logistic Regression, LightGBM, or XGBoost.",
    )
    set_paragraph(
        document,
        249,
        "The model-specific rankings agree most strongly on ROA, which ranks first for all four evaluated models. Other ranks vary materially by algorithm, so the evidence does not support a claim of complete cross-model consistency.",
    )
    set_paragraph(
        document,
        250,
        "For XGBoost and LightGBM, ROA and Current Ratio are the two leading permutation features on the retrospective holdout. Logistic Regression ranks ROA and Debt-to-Assets highest, while ANN places ROA first and Debt-to-Assets second.",
    )
    set_paragraph(
        document,
        251,
        "Because permutation estimates are based on only 12 positive holdout cases, small or negative values are treated as unstable rather than as proof that a financial indicator has no economic relevance.",
    )
    set_paragraph(
        document,
        252,
        "The rankings are predictive associations under the fitted models. They do not establish causality, and correlated ratios can redistribute permutation importance across features.",
    )
    set_paragraph(
        document,
        253,
        "Accordingly, interpretation focuses on robust recurring signals and reports the model-specific tables rather than a single universal ordering.",
    )
    set_paragraph(
        document,
        271,
        "This study evaluates machine-learning models for predicting a T+1 financial-distress proxy among Vietnamese listed companies using a temporal research design. The outcome is financial distress, not verified bankruptcy or probability of default.",
    )
    set_paragraph(
        document,
        272,
        "On the 2022–2023 retrospective holdout, XGBoost and LightGBM achieve the strongest ranking performance. XGBoost provides the more useful recall-oriented operating point under the development-selected F2 threshold. Logistic Regression remains a competitive interpretable baseline, while ANN performance is less stable due to the small number of positive events.",
    )
    set_paragraph(
        document,
        273,
        "The T+1 label is now reproducibly generated from raw statement fields. Remaining provenance work is to attach the original audited PDF, URL, page number, reporting scope, and independent verifier to every positive event. Conclusions remain conditional until this source audit and a genuinely external validation are completed.",
    )

    feature_table = document.tables[2]
    feature_table.cell(6, 1).text = "Cash and cash equivalents / Current liabilities"

    parameter_table = document.tables[3]
    parameter_rows = [
        ("Artificial Neural Network", "Epochs / Dropout / Learning rate / Weight decay", "40 / 0.20 / 0.001 / 0.00001"),
        ("Logistic Regression", "C / Penalty / Calibration", "1.0 / L2 / Sigmoid CV=3"),
        ("XGBoost", "Trees / Depth / Learning rate / Subsample / Column sample", "160 / 4 / 0.05 / 0.85 / 0.85"),
        ("LightGBM", "Trees / Depth / Learning rate / Leaves / Min child", "300 / 4 / 0.03 / 15 / 25"),
        ("Temporal Ensemble", "Base models / Meta model", "ANN+LR+LightGBM+XGBoost / Calibrated LR"),
    ]
    for row, values in zip(parameter_table.rows[1:], parameter_rows):
        for cell, value in zip(row.cells, values):
            cell.text = value

    performance_table = document.tables[4]
    performance_headers = ["Model", "AUC", "PR-AUC", "Precision", "Recall", "F2"]
    for cell, text in zip(performance_table.rows[0].cells, performance_headers):
        cell.text = text
    model_order = ["ann", "lr", "xgb", "tree"]
    model_names = {"ann": "ANN", "lr": "Logistic Regression", "xgb": "XGBoost", "tree": "LightGBM"}
    for row, model in zip(performance_table.rows[1:5], model_order):
        value = metrics(model)
        cells = [
            model_names[model],
            pct(value["AUC"]),
            pct(value["PR_AUC"]),
            pct(value["Precision"]),
            pct(value["Recall"]),
            pct(value["F2"]),
        ]
        for cell, text in zip(row.cells, cells):
            cell.text = text
    last = performance_table.rows[5]
    for cell, text in zip(last.cells, ["Temporal Ensemble", "N/A", "N/A", "N/A", "N/A", "Not evaluated"]):
        cell.text = text

    ranking_table = document.tables[5]
    headers = ["Rank", "ANN", "Logistic Regression", "XGBoost", "LightGBM", "Hybrid selection score"]
    for cell, text in zip(ranking_table.rows[0].cells, headers):
        cell.text = text
    rankings = {model: ranking(model) for model in model_order}
    hybrid = pd.read_csv(RESULTS / "tree" / "feature_influence_analysis_tree.csv")["feature"].tolist()
    for rank, row in enumerate(ranking_table.rows[1:], start=1):
        values = [str(rank)]
        for model in model_order:
            values.append(rankings[model][rank - 1] if rank <= len(rankings[model]) else "—")
        values.append(hybrid[rank - 1] if rank <= len(hybrid) else "—")
        for cell, text in zip(row.cells, values):
            cell.text = text

    # Remove stale CatBoost/Random-Forest feature-importance figures. Their values
    # came from the superseded random-split run and are not comparable to the
    # model-specific temporal-holdout permutation rankings in Table 5.
    stale_feature_figure_paragraphs = [document.paragraphs[index] for index in range(254, 265)]
    for paragraph in stale_feature_figure_paragraphs:
        remove_paragraph(paragraph)

    document.add_heading("Updated references for financial-distress labeling", level=1)
    document.add_paragraph("Vu et al. (2019). Feature selection methods and sampling techniques to financial distress prediction for Vietnamese listed companies. DOI: 10.21511/imfi.16(1).2019.22.")
    document.add_paragraph("Thinh et al. (2020). Financial distress prediction of listed companies—empirical evidence on the Vietnamese stock market. DOI: 10.21511/imfi.17(2).2020.29.")
    document.add_paragraph("Kim & Manh (2014). Modeling of Financial Distress Probability for Vietnamese Listed Companies. DOI: 10.33301/2014.16.03.04.")
    document.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
