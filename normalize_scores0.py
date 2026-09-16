import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent
OUTPUT_FILE = BASE_DIR / "normalized_all_problem_statements.xlsx"

# Any .xlsx file inside a PS folder is treated as one faculty/panel.
# The filename (without extension) becomes the Faculty name.

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
    total_column = find_column(
        df,
        [
            "total",
            "total score",
            "total (25 marks)",
            "total score (25)",
            "total_score",
        ],
    )

    if total_column is not None:
        return total_column

    # If Total does not exist, calculate it from numeric
    # scoring columns while excluding obvious non-score columns.
    excluded = set()

    for col in df.columns:
        name = str(col).strip().lower()

        if (
            "team" in name
            or "leader" in name
            or "email" in name
            or name == "name"
            or "room" in name
            or "venue" in name
            or "remark" in name
            or "comment" in name
            or "ppt" in name
            or "link" in name
            or "panel" in name
            or "faculty" in name
        ):
            excluded.add(col)

    numeric_columns = []

    for col in df.columns:
        if col in excluded:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")

        if len(df) > 0:
            percentage_numeric = numeric.notna().sum() / len(df)

            if percentage_numeric >= 0.80:
                numeric_columns.append(col)

    if not numeric_columns:
        raise ValueError(
            "Could not find Total column or scoring columns."
        )

    df["Total"] = (
        df[numeric_columns]
        .apply(pd.to_numeric, errors="coerce")
        .sum(axis=1)
    )

    return "Total"


# ============================================================
# NORMALIZE ONE FACULTY / PANEL
# ============================================================
#
# IMPORTANT:
# This is where the benchmark behavior happens.
#
# Every Excel file is normalized independently:
#   1. Panel Percentile
#   2. Panel Norm (z-score)
#   3. Min-Max Score
#
# The ranks are NOT calculated here.
# Global ranks are calculated only after every PS/faculty
# file has been combined.
# ============================================================

def normalize_faculty(df, faculty_name):
    df = df.copy()

    total_column = detect_total_column(df)

    # Preserve the original columns and add a standard Total Score.
    df["Total Score"] = pd.to_numeric(
        df[total_column],
        errors="coerce"
    )

    # Ignore completely invalid score rows.
    df = df.dropna(subset=["Total Score"]).copy()

    if len(df) == 0:
        raise ValueError(
            f"No valid scores found for {faculty_name}"
        )

    scores = df["Total Score"]

    # --------------------------------------------------------
    # PANEL PERCENTILE
    # --------------------------------------------------------
    #
    # Benchmark-compatible:
    # rank(method="max", pct=True) * 100
    #
    # This means tied highest scores both receive 100.
    # --------------------------------------------------------

    if len(df) == 1:
        df["Panel Percentile"] = 100.0
    else:
        df["Panel Percentile"] = (
            scores.rank(
                method="max",
                pct=True
            ) * 100
        )

    # --------------------------------------------------------
    # PANEL Z-SCORE
    # --------------------------------------------------------
    #
    # Benchmark uses the standard deviation of the individual
    # faculty/panel, not the combined global population.
    #
    # ddof=1 = sample standard deviation.
    # --------------------------------------------------------

    mean = scores.mean()
    std = scores.std(ddof=1)

    if pd.isna(std) or std == 0:
        df["Z Score"] = 0.0
        df["Panel Norm"] = 0.0
    else:
        df["Z Score"] = (
            scores - mean
        ) / std

        # Benchmark's panel_norm is the raw z-score.
        df["Panel Norm"] = df["Z Score"]

    # --------------------------------------------------------
    # PANEL MIN-MAX
    # --------------------------------------------------------
    #
    # Benchmark keeps this in the 0-1 range.
    # --------------------------------------------------------

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        df["Min-Max Score"] = 0.0
    else:
        df["Min-Max Score"] = (
            (scores - minimum)
            / (maximum - minimum)
        )

    # Faculty/file name
    df["Faculty"] = faculty_name

    statistics = {
        "Faculty": faculty_name,
        "Team Count": len(df),
        "Mean Total": mean,
        "SD": std,
        "Minimum Total": minimum,
        "Maximum Total": maximum,
    }

    return df, statistics


# ============================================================
# FIND ALL FACULTY FILES IN ONE PS FOLDER
# ============================================================

def find_faculty_files(ps_folder):
    files = []

    for file in ps_folder.iterdir():
        if not file.is_file():
            continue

        if file.suffix.lower() != ".xlsx":
            continue

        # Never treat the generated output as an input.
        if file.name.lower() == OUTPUT_FILE.name.lower():
            continue

        # Faculty name = Excel filename without extension.
        files.append(file)

    return sorted(
        files,
        key=lambda x: x.stem.lower()
    )


# ============================================================
# PROCESS ONE PROBLEM STATEMENT
# ============================================================

def process_problem_statement(ps_folder):
    faculty_files = find_faculty_files(ps_folder)

    if not faculty_files:
        return None, None

    print(f"\nProcessing: {ps_folder.name}")

    all_faculty_results = []
    all_faculty_statistics = []

    for faculty_file in faculty_files:
        faculty_name = faculty_file.stem.strip()

        print(f"  Faculty: {faculty_file.name}")

        df = pd.read_excel(faculty_file)

        result, statistics = normalize_faculty(
            df,
            faculty_name
        )

        # Add PS and source metadata.
        result.insert(
            0,
            "Problem Statement",
            ps_folder.name
        )

        result["Source File"] = faculty_file.name

        all_faculty_results.append(result)

        all_faculty_statistics.append(
            {
                "Problem Statement": ps_folder.name,
                **statistics
            }
        )

    combined = pd.concat(
        all_faculty_results,
        ignore_index=True
    )

    statistics = pd.DataFrame(
        all_faculty_statistics
    )

    return combined, statistics


# ============================================================
# CALCULATE GLOBAL RANKS
# ============================================================
#
# At this point every team from every PS and every faculty
# file is in one dataframe.
#
# The normalized values themselves were calculated per faculty.
# These three normalized columns are now ranked globally.
# ============================================================

def calculate_global_ranks(all_combined):
    df = all_combined.copy()

    # Higher normalized value = better rank.
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

    # Equal weight for all three global ranking systems.
    df["Avg Rank"] = (
        df["Z Rank"]
        + df["Pct Rank"]
        + df["MinMax Rank"]
    ) / 3

    # --------------------------------------------------------
    # Global ordering
    # --------------------------------------------------------

    df = df.sort_values(
        by=[
            "Avg Rank",
            "Total Score"
        ],
        ascending=[
            True,
            False
        ]
    ).reset_index(drop=True)

    # Sequential final rank after global sorting.
    df["Final Rank"] = (
        df.index + 1
    )

    # --------------------------------------------------------
    # Near Tie
    # --------------------------------------------------------

    df["Near Tie"] = ""

    for i in range(len(df) - 1):
        current_rank = df.loc[
            i,
            "Avg Rank"
        ]

        next_rank = df.loc[
            i + 1,
            "Avg Rank"
        ]

        difference = abs(
            current_rank - next_rank
        )

        if difference <= NEAR_TIE_RANK_DIFFERENCE:
            df.loc[
                i,
                "Near Tie"
            ] = "YES"

            df.loc[
                i + 1,
                "Near Tie"
            ] = "YES"

    return df


# ============================================================
# FORMAT EXCEL
# ============================================================

def format_excel(output_file):
    from openpyxl import load_workbook

    workbook = load_workbook(output_file)

    for worksheet in workbook.worksheets:

        # Freeze header
        worksheet.freeze_panes = "A2"

        # Autofilter
        worksheet.auto_filter.ref = (
            worksheet.dimensions
        )

        # Dynamic column widths
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

    workbook.save(output_file)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BENCHMARK-STYLE GLOBAL SCORE NORMALIZATION")
    print("=" * 60)

    all_results = []
    all_statistics = []

    # --------------------------------------------------------
    # Find PS folders dynamically.
    # --------------------------------------------------------

    ps_folders = []

    for folder in BASE_DIR.iterdir():

        if not folder.is_dir():
            continue

        # Ignore hidden folders such as .claude / .kilo
        if folder.name.startswith("."):
            continue

        faculty_files = find_faculty_files(folder)

        if faculty_files:
            ps_folders.append(folder)

    # --------------------------------------------------------
    # Nothing found
    # --------------------------------------------------------

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
            "\n│   ├── faculty2.xlsx"
            "\n│   └── faculty3.xlsx"
            "\n├── PS02/"
            "\n│   ├── faculty1.xlsx"
            "\n│   └── faculty2.xlsx"
            "\n└── ..."
        )

        return

    # --------------------------------------------------------
    # Process every PS.
    # --------------------------------------------------------

    for ps_folder in sorted(
        ps_folders,
        key=lambda x: x.name.lower()
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

    if not all_results:
        print("\nNo usable data found.")
        return

    # --------------------------------------------------------
    # Combine EVERYTHING globally.
    # --------------------------------------------------------

    all_combined = pd.concat(
        all_results,
        ignore_index=True
    )

    # --------------------------------------------------------
    # GLOBAL RANKING
    # --------------------------------------------------------

    all_combined = calculate_global_ranks(
        all_combined
    )

    # --------------------------------------------------------
    # Global statistics
    # --------------------------------------------------------

    all_stats = pd.concat(
        all_statistics,
        ignore_index=True
    )

    global_statistics = pd.DataFrame(
        [
            {
                "Metric": "Total Teams",
                "Value": len(all_combined)
            },
            {
                "Metric": "Problem Statements",
                "Value": all_combined[
                    "Problem Statement"
                ].nunique()
            },
            {
                "Metric": "Faculty Files",
                "Value": all_combined[
                    ["Problem Statement", "Faculty"]
                ].drop_duplicates().shape[0]
            },
        ]
    )

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------

    with pd.ExcelWriter(
        OUTPUT_FILE,
        engine="openpyxl"
    ) as writer:

        # One global ranking — this is the important sheet.
        all_combined.to_excel(
            writer,
            sheet_name="All Rankings",
            index=False
        )

        # Statistics used by each faculty normalization.
        all_stats.to_excel(
            writer,
            sheet_name="Faculty Statistics",
            index=False
        )

        # Descriptive global information.
        global_statistics.to_excel(
            writer,
            sheet_name="Global Statistics",
            index=False
        )

    # --------------------------------------------------------
    # Format workbook
    # --------------------------------------------------------

    format_excel(OUTPUT_FILE)

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("NORMALIZATION COMPLETED")
    print("=" * 60)

    print(
        f"\nProblem Statements processed: "
        f"{len(ps_folders)}"
    )

    print(
        f"Total teams ranked globally: "
        f"{len(all_combined)}"
    )

    print(
        f"\nOutput:"
    )

    print(OUTPUT_FILE)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
