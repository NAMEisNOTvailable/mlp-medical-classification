from pathlib import Path


def test_readme_links_portfolio_assets():
    readme = Path("README.md").read_text(encoding="utf-8")

    expected_paths = [
        "assets/model_selection_overview.png",
        "docs/portfolio_summary.md",
        "results/model_comparison.csv",
        "results/summary.json",
        "results/auc_comparison.png",
        "results/precision_recall_curves.png",
    ]
    for path in expected_paths:
        assert path in readme
        assert Path(path).exists()


def test_generated_overview_chart_is_committed():
    chart_path = Path("assets/model_selection_overview.png")

    assert chart_path.exists()
    assert chart_path.stat().st_size > 10_000
