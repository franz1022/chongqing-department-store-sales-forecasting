from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ============================================================
# 0. 路径设置
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = BASE_DIR / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = FIGURE_DIR / "17_project_workflow_diagram.png"


# ============================================================
# 1. 可修改参数
# ============================================================

FIG_W = 24
FIG_H = 14
DPI = 180

TITLE = "Retail Sales Forecasting Project Workflow"
SUBTITLE = (
    "Data validation, leakage-safe modeling, "
    "feature availability audit, and rolling-origin backtesting"
)

BOX_W = 3.55
BOX_H = 1.25
BOX_ROUND = 0.08

FONT_TITLE = 23
FONT_SUBTITLE = 12
FONT_BOX = 9
FONT_SECTION = 13
FONT_FOOTER = 10

BOX_LW = 1.5
ARROW_LW = 1.8

COLOR_RAW = "#DCEBFA"
COLOR_VALIDATION = "#D9EAD3"
COLOR_ANALYSIS = "#FFF2CC"
COLOR_MODEL = "#FCE5CD"
COLOR_EVALUATION = "#EADCF8"
COLOR_FINAL = "#F4CCCC"
COLOR_EDGE = "#444444"

X_POSITIONS = [
    0.7,
    5.1,
    9.5,
    13.9,
    18.3,
]

ROW_1_Y = 10.0
ROW_2_Y = 6.2
ROW_3_Y = 2.4


# ============================================================
# 2. 画布
# ============================================================

fig, ax = plt.subplots(
    figsize=(FIG_W, FIG_H),
    dpi=DPI,
)

ax.set_xlim(0, 22.6)
ax.set_ylim(0, 14)
ax.axis("off")


# ============================================================
# 3. 工具函数
# ============================================================

def draw_box(
    x,
    y,
    text,
    facecolor,
    fontsize=FONT_BOX,
):
    box = FancyBboxPatch(
        (x, y),
        BOX_W,
        BOX_H,
        boxstyle=(
            "round,pad=0.02,"
            f"rounding_size={BOX_ROUND}"
        ),
        linewidth=BOX_LW,
        edgecolor=COLOR_EDGE,
        facecolor=facecolor,
    )

    ax.add_patch(box)

    ax.text(
        x + BOX_W / 2,
        y + BOX_H / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def draw_horizontal_arrow(
    start_x,
    end_x,
    y,
):
    if end_x > start_x:
        point_1 = (
            start_x + BOX_W,
            y + BOX_H / 2,
        )
        point_2 = (
            end_x,
            y + BOX_H / 2,
        )
    else:
        point_1 = (
            start_x,
            y + BOX_H / 2,
        )
        point_2 = (
            end_x + BOX_W,
            y + BOX_H / 2,
        )

    arrow = FancyArrowPatch(
        point_1,
        point_2,
        arrowstyle="->",
        mutation_scale=15,
        linewidth=ARROW_LW,
        color=COLOR_EDGE,
    )

    ax.add_patch(arrow)


def draw_vertical_arrow(
    x,
    start_y,
    end_y,
):
    point_1 = (
        x + BOX_W / 2,
        start_y,
    )

    point_2 = (
        x + BOX_W / 2,
        end_y + BOX_H,
    )

    arrow = FancyArrowPatch(
        point_1,
        point_2,
        arrowstyle="->",
        mutation_scale=15,
        linewidth=ARROW_LW,
        color=COLOR_EDGE,
    )

    ax.add_patch(arrow)


def draw_section_title(
    x,
    y,
    text,
):
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="bottom",
        fontsize=FONT_SECTION,
        fontweight="bold",
    )


# ============================================================
# 4. 标题
# ============================================================

ax.text(
    11.3,
    13.25,
    TITLE,
    ha="center",
    va="center",
    fontsize=FONT_TITLE,
    fontweight="bold",
)

ax.text(
    11.3,
    12.72,
    SUBTITLE,
    ha="center",
    va="center",
    fontsize=FONT_SUBTITLE,
)


# ============================================================
# 5. 第一层：数据验证与清洗
# ============================================================

draw_section_title(
    0.7,
    11.45,
    "1. Data Validation and Preparation",
)

draw_box(
    X_POSITIONS[0],
    ROW_1_Y,
    "Raw Tables\nsales / features /\nstores",
    COLOR_RAW,
)

draw_box(
    X_POSITIONS[1],
    ROW_1_Y,
    "00_inspect_raw_data.py\nRaw Structure\nInspection",
    COLOR_VALIDATION,
)

draw_box(
    X_POSITIONS[2],
    ROW_1_Y,
    "00_validate_relationships.py\nBusiness Keys &\nRelationship Checks",
    COLOR_VALIDATION,
)

draw_box(
    X_POSITIONS[3],
    ROW_1_Y,
    "01_data_cleaning.py\nSafe Cleaning &\nMany-to-One Merge",
    COLOR_VALIDATION,
)

draw_box(
    X_POSITIONS[4],
    ROW_1_Y,
    "00_audit_cleaned_data.py\nPost-Cleaning\nIntegrity Audit",
    COLOR_VALIDATION,
)

for index in range(4):
    draw_horizontal_arrow(
        X_POSITIONS[index],
        X_POSITIONS[index + 1],
        ROW_1_Y,
    )


# ============================================================
# 6. 第二层：分析与模型开发
# 流程方向为从右向左，避免跨层斜线交叉
# ============================================================

draw_section_title(
    0.7,
    7.65,
    "2. Analysis, Feature Engineering, and Modeling",
)

draw_box(
    X_POSITIONS[4],
    ROW_2_Y,
    "02_summer_eda.py\nSummer EDA &\nBusiness Patterns",
    COLOR_ANALYSIS,
)

draw_box(
    X_POSITIONS[3],
    ROW_2_Y,
    "Leakage-Safe Features\nLag / Rolling /\nCalendar / Markdown",
    COLOR_ANALYSIS,
)

draw_box(
    X_POSITIONS[2],
    ROW_2_Y,
    "03 + 04 Models\nBaselines / Rolling\nARIMA & SARIMA",
    COLOR_MODEL,
)

draw_box(
    X_POSITIONS[1],
    ROW_2_Y,
    "05 ML Models\nLinear / Random Forest /\nXGBoost",
    COLOR_MODEL,
)

draw_box(
    X_POSITIONS[0],
    ROW_2_Y,
    "05b Feature Audit\nFull vs Operational\n27 vs 23 Features",
    COLOR_EVALUATION,
)

draw_vertical_arrow(
    X_POSITIONS[4],
    ROW_1_Y,
    ROW_2_Y,
)

for index in range(4, 0, -1):
    draw_horizontal_arrow(
        X_POSITIONS[index],
        X_POSITIONS[index - 1],
        ROW_2_Y,
    )


# ============================================================
# 7. 第三层：评估、模型选择与交付
# ============================================================

draw_section_title(
    0.7,
    3.85,
    "3. Evaluation, Model Selection, and Business Delivery",
)

draw_box(
    X_POSITIONS[0],
    ROW_3_Y,
    "Single-Window Evaluation\nGranular vs Company\nMAE / RMSE / WAPE",
    COLOR_EVALUATION,
)

draw_box(
    X_POSITIONS[1],
    ROW_3_Y,
    "06 Rolling-Origin\n6 Expanding Folds\n14 Weeks per Fold",
    COLOR_EVALUATION,
)

draw_box(
    X_POSITIONS[2],
    ROW_3_Y,
    "Stability Review\nMean / Median / Std\nBias / Fold Wins",
    COLOR_EVALUATION,
)

draw_box(
    X_POSITIONS[3],
    ROW_3_Y,
    "Final Model\nOperational XGBoost\nMean WAPE 1.91%",
    COLOR_FINAL,
)

draw_box(
    X_POSITIONS[4],
    ROW_3_Y,
    "Business Delivery\nRecommendations /\nREADME / Figures",
    COLOR_FINAL,
)

draw_vertical_arrow(
    X_POSITIONS[0],
    ROW_2_Y,
    ROW_3_Y,
)

for index in range(4):
    draw_horizontal_arrow(
        X_POSITIONS[index],
        X_POSITIONS[index + 1],
        ROW_3_Y,
    )


# ============================================================
# 8. 页脚
# ============================================================

footer_text = (
    "50 stores | 20 departments | 156 weeks | "
    "156,000 rows | rolling one-week-ahead forecasting | "
    "Operational XGBoost won 5 of 6 folds"
)

ax.text(
    11.3,
    0.75,
    footer_text,
    ha="center",
    va="center",
    fontsize=FONT_FOOTER,
)


# ============================================================
# 9. 保存
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    bbox_inches="tight",
)

plt.close()

print(
    "Updated project workflow diagram saved to:"
)

print(OUTPUT_PATH)
