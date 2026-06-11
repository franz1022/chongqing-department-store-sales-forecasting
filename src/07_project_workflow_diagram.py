import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ============================================================
# 0. 路径设置
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = BASE_DIR / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = FIGURE_DIR / "15_project_workflow_diagram.png"


# ============================================================
# 1. 可修改区域（你以后想调图，主要改这里）
# ============================================================

FIG_W = 18
FIG_H = 10
DPI = 180

TITLE = "Retail Sales Forecasting Project Workflow"
SUBTITLE = "Chongqing Department Store Summer Sales Forecasting"

BOX_W = 2.8
BOX_H = 1.0
BOX_ROUND = 0.08

FONT_TITLE = 22
FONT_SUBTITLE = 12
FONT_BOX = 10
FONT_SECTION = 13

ARROW_LW = 1.8
BOX_LW = 1.5

# 颜色
COLOR_DATA = "#DCEBFA"
COLOR_EDA = "#E8F6E8"
COLOR_MODEL = "#FFF2CC"
COLOR_EVAL = "#FCE4EC"
COLOR_BIZ = "#EDE7F6"
COLOR_EDGE = "#4A4A4A"


# ============================================================
# 2. 画布
# ============================================================

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis("off")


# ============================================================
# 3. 工具函数
# ============================================================

def draw_box(x, y, w, h, text, facecolor, fontsize=FONT_BOX):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={BOX_ROUND}",
        linewidth=BOX_LW,
        edgecolor=COLOR_EDGE,
        facecolor=facecolor
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize
    )


def draw_arrow(x1, y1, x2, y2):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="->",
        mutation_scale=14,
        linewidth=ARROW_LW,
        color=COLOR_EDGE
    )
    ax.add_patch(arrow)


def draw_section_title(x, y, text):
    ax.text(
        x, y, text,
        ha="left",
        va="bottom",
        fontsize=FONT_SECTION,
        fontweight="bold"
    )


# ============================================================
# 4. 标题
# ============================================================

ax.text(
    10, 11.3,
    TITLE,
    ha="center", va="center",
    fontsize=FONT_TITLE,
    fontweight="bold"
)

ax.text(
    10, 10.8,
    SUBTITLE,
    ha="center", va="center",
    fontsize=FONT_SUBTITLE
)


# ============================================================
# 5. Section Titles
# ============================================================

draw_section_title(0.8, 9.8, "Data Layer")
draw_section_title(0.8, 7.0, "Analysis & Modeling Layer")
draw_section_title(0.8, 3.2, "Evaluation & Business Layer")


# ============================================================
# 6. 第一层：数据层
# ============================================================

draw_box(1.0, 8.5, BOX_W, BOX_H, "sales.csv\nWeekly Sales", COLOR_DATA)
draw_box(4.2, 8.5, BOX_W, BOX_H, "features.csv\nMarkdown / Holiday /\nEconomic Features", COLOR_DATA)
draw_box(7.4, 8.5, BOX_W, BOX_H, "stores.csv\nStore Type / Size /\nRegion", COLOR_DATA)
draw_box(10.8, 8.5, BOX_W, BOX_H, "01_data_cleaning.py\nData Cleaning & Merge", COLOR_DATA)
draw_box(14.2, 8.5, BOX_W, BOX_H, "Merged Dataset\nretail_sales_cleaned_merged.csv", COLOR_DATA)

draw_arrow(3.8, 9.0, 4.2, 9.0)
draw_arrow(7.0, 9.0, 7.4, 9.0)
draw_arrow(10.2, 9.0, 10.8, 9.0)
draw_arrow(13.6, 9.0, 14.2, 9.0)


# ============================================================
# 7. 第二层：分析与建模层
# ============================================================

draw_box(1.0, 5.7, BOX_W, BOX_H, "02_summer_eda.py\nEDA & Summer Dataset", COLOR_EDA)
draw_box(4.2, 5.7, BOX_W, BOX_H, "03_baseline_forecasting.py\nLast Week / Rolling Avg /\nPrev-Year Baseline", COLOR_MODEL)
draw_box(7.4, 5.7, BOX_W, BOX_H, "04_arima_sarima.py\nARIMA / SARIMA", COLOR_MODEL)
draw_box(10.8, 5.7, BOX_W, BOX_H, "05_ml_regression_models.py\nLinear Regression /\nRandom Forest / XGBoost", COLOR_MODEL)
draw_box(14.2, 5.7, BOX_W, BOX_H, "Feature Engineering\nLag / Rolling /\nMarkdown / Holiday", COLOR_MODEL)

# 数据层向下
draw_arrow(15.6, 8.5, 2.4, 6.7)
draw_arrow(15.6, 8.5, 5.6, 6.7)
draw_arrow(15.6, 8.5, 8.8, 6.7)
draw_arrow(15.6, 8.5, 12.2, 6.7)
draw_arrow(15.6, 8.5, 15.6, 6.7)

# 横向关系
draw_arrow(3.8, 6.2, 4.2, 6.2)
draw_arrow(7.0, 6.2, 7.4, 6.2)
draw_arrow(10.2, 6.2, 10.8, 6.2)
draw_arrow(13.6, 6.2, 14.2, 6.2)


# ============================================================
# 8. 第三层：评估与业务层
# ============================================================

draw_box(1.0, 2.0, 3.0, 1.2, "Aggregate-Level Evaluation\nMAE / RMSE / MAPE", COLOR_EVAL)
draw_box(4.8, 2.0, 3.0, 1.2, "Model Comparison\nBaseline vs ARIMA vs ML", COLOR_EVAL)
draw_box(8.6, 2.0, 3.0, 1.2, "Feature Importance\nXGBoost Insights", COLOR_EVAL)
draw_box(12.4, 2.0, 3.0, 1.2, "06_business_recommendations.py\nBusiness Recommendations", COLOR_BIZ)
draw_box(16.0, 2.0, 3.0, 1.2, "Final Outputs\nREADME / Figures /\nSummary Metrics", COLOR_BIZ)

# 第二层到第三层
draw_arrow(2.4, 5.7, 2.5, 3.2)
draw_arrow(5.6, 5.7, 6.2, 3.2)
draw_arrow(8.8, 5.7, 9.8, 3.2)
draw_arrow(12.2, 5.7, 13.8, 3.2)
draw_arrow(15.6, 5.7, 17.5, 3.2)

# 第三层横向
draw_arrow(4.0, 2.6, 4.8, 2.6)
draw_arrow(7.8, 2.6, 8.6, 2.6)
draw_arrow(11.6, 2.6, 12.4, 2.6)
draw_arrow(15.4, 2.6, 16.0, 2.6)


# ============================================================
# 9. 页脚说明
# ============================================================

footer_text = (
    "Workflow Summary: Raw retail data → cleaning & merging → EDA → baseline/statistical/ML forecasting "
    "→ aggregate evaluation → business recommendations."
)

ax.text(
    10, 0.7,
    footer_text,
    ha="center", va="center",
    fontsize=10
)


# ============================================================
# 10. 保存
# ============================================================

plt.tight_layout()
plt.savefig(OUTPUT_PATH, bbox_inches="tight")
plt.close()

print(f"Project workflow diagram saved to: {OUTPUT_PATH}")