import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent

OUTPUT_FILE = BASE_DIR / "normalized_all_problem_statements.xlsx"

# Files expected inside each PS folder
FACULTY1_NAME = "faculty1.xlsx"
FACULTY2_NAME = "faculty2.xlsx"

# Near-tie threshold for Avg Rank
# Example: Rank 4.00 and Rank 4.33 -> near tie
NEAR_TIE_RANK_DIFFERENCE = 0.50


# ============================================================
# FIND COLUMN DYNAMICALLY
# ============================================================

def find_column(df, possible_names):

    for column in df.columns:

        clean_name = str(column).strip().lower()

        for name in possible_names:

            if clean_name == name:
                return column

    return None


# ============================================================
# DETECT TOTAL COLUMN
# ============================================================

def detect_total_column(df):

    # First look for an actual Total column
    total_column = find_column(
        df,
        [
            "total",
            "total score",
            "total (25 marks)",
            "total score (25)"
        ]
    )

    if total_column is not None:
        return total_column

    # If Total doesn't exist, try to calculate it
    # from numeric columns.
    excluded = set()

    for col in df.columns:

        name = str(col).strip().lower()

        if (
            "team" in name
            or "leader" in name
            or "name" == name
            or "room" in name
            or "remark" in name
            or "comment" in name
            or "ppt" in name
            or "link" in name
        ):
            excluded.add(col)

    numeric_columns = []

    for col in df.columns:

        if col in excluded:
            continue

        numeric = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if len(df) > 0:

            percentage_numeric = (
                numeric.notna().sum() / len(df)
            )

            if percentage_numeric >= 0.80:
                numeric_columns.append(col)

    if not numeric_columns:

        raise ValueError(
            "Could not find Total column or scoring columns."
        )

    # Calculate Total automatically
    df["Total"] = df[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce"
    ).sum(axis=1)

    return "Total"


# ============================================================
# NORMALIZE ONE FACULTY
# ============================================================

def normalize_faculty(df, faculty_name):

    df = df.copy()

    # --------------------------------------------------------
    # Detect Total dynamically
    # --------------------------------------------------------

    total_column = detect_total_column(df)

    # Create a standard output column while preserving
    # the original Total column.
    df["Total Score"] = pd.to_numeric(
        df[total_column],
        errors="coerce"
    )

    # Remove completely empty rows
    df = df.dropna(
        subset=["Total Score"]
    ).copy()

    if len(df) == 0:

        raise ValueError(
            f"No valid scores found for {faculty_name}"
        )

    scores = df["Total Score"]

    # ========================================================
    # PANEL PERCENTILE
    # ========================================================

    if len(df) == 1:

        df["Panel Percentile"] = 100.0

    else:

        df["Panel Percentile"] = (
            scores.rank(
                method="average",
                pct=True
            ) * 100
        )

    # ========================================================
    # Z-SCORE
    # ========================================================

    mean = scores.mean()

    std = scores.std(ddof=0)

    if std == 0:

        df["Z Score"] = 0.0

        df["Panel Norm"] = 50.0

    else:

        df["Z Score"] = (
            scores - mean
        ) / std

        # Convert z-score to 0-100
        df["Panel Norm"] = (
            norm.cdf(
                df["Z Score"]
            ) * 100
        )

    # ========================================================
    # MIN-MAX
    # ========================================================

    minimum = scores.min()

    maximum = scores.max()

    if maximum == minimum:

        df["Min-Max Score"] = 50.0

    else:

        df["Min-Max Score"] = (
            (scores - minimum)
            / (maximum - minimum)
        ) * 100

    # ========================================================
    # RANKS
    # ========================================================

    # Higher normalized score = better rank

    df["Z Rank"] = (
        df["Panel Norm"]
        .rank(
            ascending=False,
            method="average"
        )
    )

    df["Pct Rank"] = (
        df["Panel Percentile"]
        .rank(
            ascending=False,
            method="average"
        )
    )

    df["MinMax Rank"] = (
        df["Min-Max Score"]
        .rank(
            ascending=False,
            method="average"
        )
    )

    # ========================================================
    # EQUAL-WEIGHT FINAL RANK
    # ========================================================

    df["Avg Rank"] = (
        df["Z Rank"]
        + df["Pct Rank"]
        + df["MinMax Rank"]
    ) / 3

    # ========================================================
    # FACULTY
    # ========================================================

    df["Faculty"] = faculty_name

    # Statistics
    statistics = {
        "Faculty": faculty_name,
        "Team Count": len(df),
        "Mean Total": mean,
        "SD": std,
        "Minimum Total": minimum,
        "Maximum Total": maximum
    }

    return df, statistics


# ============================================================
# FIND FACULTY FILES
# ============================================================

def find_faculty_file(folder, faculty_number):

    target = f"faculty{faculty_number}"

    # Case-insensitive search
    for file in folder.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() != ".xlsx":
            continue

        stem = file.stem.strip().lower()

        if stem == target:
            return file

    return None


# ============================================================
# PROCESS ONE PROBLEM STATEMENT
# ============================================================

def process_problem_statement(ps_folder):

    faculty1_file = find_faculty_file(
        ps_folder,
        1
    )

    faculty2_file = find_faculty_file(
        ps_folder,
        2
    )

    # Don't process random folders
    if faculty1_file is None or faculty2_file is None:

        return None, None

    print(
        f"\nProcessing: {ps_folder.name}"
    )

    print(
        f"  Faculty 1: {faculty1_file.name}"
    )

    print(
        f"  Faculty 2: {faculty2_file.name}"
    )

    # --------------------------------------------------------
    # Read Excel
    # --------------------------------------------------------

    faculty1_df = pd.read_excel(
        faculty1_file
    )

    faculty2_df = pd.read_excel(
        faculty2_file
    )

    # --------------------------------------------------------
    # Normalize separately
    # --------------------------------------------------------

    faculty1_result, stats1 = normalize_faculty(
        faculty1_df,
        "Faculty 1"
    )

    faculty2_result, stats2 = normalize_faculty(
        faculty2_df,
        "Faculty 2"
    )

    # --------------------------------------------------------
    # Add PS
    # --------------------------------------------------------

    faculty1_result.insert(
        0,
        "Problem Statement",
        ps_folder.name
    )

    faculty2_result.insert(
        0,
        "Problem Statement",
        ps_folder.name
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = pd.concat(
        [
            faculty1_result,
            faculty2_result
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Sort by Avg Rank
    # --------------------------------------------------------

    combined = combined.sort_values(
        by=[
            "Avg Rank",
            "Total Score"
        ],
        ascending=[
            True,
            False
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Overall Rank
    # --------------------------------------------------------

    combined["Final Rank"] = (
        combined.index + 1
    )

    # --------------------------------------------------------
    # Near Tie
    # --------------------------------------------------------

    combined["Near Tie"] = ""

    for i in range(
        len(combined) - 1
    ):

        current_rank = combined.loc[
            i,
            "Avg Rank"
        ]

        next_rank = combined.loc[
            i + 1,
            "Avg Rank"
        ]

        difference = abs(
            current_rank - next_rank
        )

        if difference <= NEAR_TIE_RANK_DIFFERENCE:

            combined.loc[
                i,
                "Near Tie"
            ] = "YES"

            combined.loc[
                i + 1,
                "Near Tie"
            ] = "YES"

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    statistics = pd.DataFrame(
        [
            {
                "Problem Statement": ps_folder.name,
                **stats1
            },
            {
                "Problem Statement": ps_folder.name,
                **stats2
            }
        ]
    )

    return combined, statistics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "DYNAMIC FACULTY SCORE NORMALIZATION"
    )
    print("=" * 60)

    all_results = []

    all_statistics = []

    # ========================================================
    # FIND ONLY REAL PS FOLDERS
    # ========================================================

    ps_folders = []

    for folder in BASE_DIR.iterdir():

        # Must be a directory
        if not folder.is_dir():
            continue

        # Ignore hidden folders such as:
        # .claude
        # .kilo
        if folder.name.startswith("."):
            continue

        # Only accept folders containing BOTH faculty files
        faculty1 = find_faculty_file(
            folder,
            1
        )

        faculty2 = find_faculty_file(
            folder,
            2
        )

        if (
            faculty1 is not None
            and faculty2 is not None
        ):

            ps_folders.append(folder)

    # ========================================================
    # NOTHING FOUND
    # ========================================================

    if not ps_folders:

        print(
            "\nNo valid Problem Statement folders found."
        )

        print(
            "\nExpected structure:"
        )

        print(
            "\nSH-Result/"
            "\n│"
            "\n├── normalize_scores.py"
            "\n├── PS01/"
            "\n│   ├── faculty1.xlsx"
            "\n│   └── faculty2.xlsx"
            "\n├── PS02/"
            "\n│   ├── faculty1.xlsx"
            "\n│   └── faculty2.xlsx"
            "\n└── PS03/"
            "\n    ├── faculty1.xlsx"
            "\n    └── faculty2.xlsx"
        )

        return

    # ========================================================
    # PROCESS ALL PS
    # ========================================================

    for ps_folder in sorted(
        ps_folders,
        key=lambda x: x.name
    ):

        result, statistics = (
            process_problem_statement(
                ps_folder
            )
        )

        if result is not None:

            all_results.append(
                result
            )

            all_statistics.append(
                statistics
            )

    # ========================================================
    # CREATE OUTPUT
    # ========================================================

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:

        # ----------------------------------------------------
        # ALL PROBLEM STATEMENTS
        # ----------------------------------------------------

        all_combined = pd.concat(
            all_results,
            ignore_index=True
        )

        all_combined.to_excel(
            writer,
            sheet_name="All Rankings",
            index=False
        )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        all_stats = pd.concat(
            all_statistics,
            ignore_index=True
        )

        all_stats.to_excel(
            writer,
            sheet_name="Faculty Statistics",
            index=False
        )

        # ----------------------------------------------------
        # ONE SHEET PER PS
        # ----------------------------------------------------

        for result in all_results:

            ps_name = str(
                result[
                    "Problem Statement"
                ].iloc[0]
            )

            # Excel sheet names max = 31 chars
            sheet_name = ps_name[:31]

            result.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )

    # ========================================================
    # FORMAT EXCEL
    # ========================================================

    from openpyxl import load_workbook

    workbook = load_workbook(
        OUTPUT_FILE
    )

    for worksheet in workbook.worksheets:

        # Freeze header
        worksheet.freeze_panes = "A2"

        # Filter
        worksheet.auto_filter.ref = (
            worksheet.dimensions
        )

        # Column widths
        for column in worksheet.columns:

            max_length = 0

            column_letter = (
                column[0].column_letter
            )

            for cell in column:

                try:

                    length = len(
                        str(cell.value)
                    )

                    max_length = max(
                        max_length,
                        length
                    )

                except Exception:
                    pass

            worksheet.column_dimensions[
                column_letter
            ].width = min(
                max(max_length + 2, 10),
                40
            )

    workbook.save(
        OUTPUT_FILE
    )

    # ========================================================
    # DONE
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "NORMALIZATION COMPLETED"
    )

    print("=" * 60)

    print(
        f"\nProblem Statements processed: "
        f"{len(all_results)}"
    )

    print(
        f"\nOutput:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()