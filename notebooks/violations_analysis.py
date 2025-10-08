import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from core.datasets.pfdelta_dataset import PFDeltaDataset
    from notebooks.utils import check_violations, compute_pct_violations
    import pandas as pd
    import altair as alt
    import matplotlib.pyplot as plt
    return (
        PFDeltaDataset,
        alt,
        check_violations,
        compute_pct_violations,
        mo,
        pd,
        plt,
    )


@app.cell
def _():
    cases_characteristics = {
        "case14":  {"n_gens": 5,   "n_buses": 14,   "n_branches": 20},
        "case30":  {"n_gens": 6,   "n_buses": 30,   "n_branches": 41},
        "case57":  {"n_gens": 7,   "n_buses": 57,   "n_branches": 80},
        "case118": {"n_gens": 54,  "n_buses": 118,  "n_branches": 186},
        "case500": {"n_gens": 99,  "n_buses": 500,  "n_branches": 597},
        "case2000": {"n_gens": 422, "n_buses": 2000, "n_branches": 3206},
    }
    return (cases_characteristics,)


@app.cell
def _(PFDeltaDataset):
    case_name = "case118"
    dataset = PFDeltaDataset(
        case_name=case_name,
        task="analysis",
        feasibility_type="feasible",
        perturbation="n"
    )
    return case_name, dataset


@app.cell
def _(check_violations, dataset, pd):
    violations_results = check_violations(dataset)
    violations_df = pd.DataFrame(violations_results)
    return (violations_df,)


@app.cell
def _(violations_df):
    violations_df
    return


@app.cell
def _(mo, violations_df):
    violation_types = ["pg", "qg", "vm", "f"]
    type_max = {
        t: int(violations_df[t].max()) if t in violations_df.columns else 0
        for t in violation_types
    }
    # give a little headroom so you can slide beyond current max
    type_max = {t: max(50, m) for t, m in type_max.items()}

    sl_pg = mo.ui.slider(0, type_max["pg"], step=1, value=min(2, type_max["pg"]), show_value=True, label="pg ≥ x")
    sl_qg = mo.ui.slider(0, type_max["qg"], step=1, value=min(5, type_max["qg"]), show_value=True, label="qg ≥ x")
    sl_vm = mo.ui.slider(0, type_max["vm"], step=1, value=min(9, type_max["vm"]), show_value=True, label="vm ≥ x")
    sl_f  = mo.ui.slider(0, type_max["f"],  step=1, value=min(10, type_max["f"]), show_value=True, label="f ≥ x")
    return sl_f, sl_pg, sl_qg, sl_vm


@app.cell
def _(
    alt,
    compute_pct_violations,
    mo,
    sl_f,
    sl_pg,
    sl_qg,
    sl_vm,
    violations_df,
):
    def _():
        thresholds = {
            "pg": sl_pg.value,
            "qg": sl_qg.value,
            "vm": sl_vm.value,
            "f":  sl_f.value,
        }
        out = compute_pct_violations(violations_df, thresholds)

        # label like "pg (≥2)" on x-axis
        out["label"] = out.apply(lambda r: f"{r['type']} (≥{int(r['threshold'])})", axis=1)

        chart = (
            alt.Chart(out)
            .mark_bar()
            .encode(
                x=alt.X("label:N", title="Violation type (threshold)"),
                y=alt.Y("percent:Q", title="% of samples ≥ threshold", scale=alt.Scale(domain=[0, 100])),
                tooltip=[
                    alt.Tooltip("type:N"),
                    alt.Tooltip("threshold:Q"),
                    alt.Tooltip("percent:Q", format=".1f"),
                ],
            )
            .properties(width=480, height=320, title="Exceedance percentages")
        ) + alt.Chart(out).mark_text(dy=-5).encode(x="label:N", y="percent:Q", text=alt.Text("percent:Q", format=".1f"))

        controls = mo.vstack([sl_pg, sl_qg, sl_vm, sl_f])
        return mo.hstack([controls, chart], align="start")
    _()
    return


@app.cell
def _(
    case_name,
    compute_pct_violations,
    make_thresholds_for_case,
    plot_pct_violations,
    violations_df,
):
    component_percentages = [25, 50, 75]
    for pct in component_percentages:
        curr_threshold = make_thresholds_for_case(case_name, pct)
        df_pct_violations = compute_pct_violations(violations_df, curr_threshold)
        plot_pct_violations(df_pct_violations)
    return


@app.cell
def _(cases_characteristics):
    def make_thresholds_for_case(case: str, pct: float):
        vals = cases_characteristics[case]
        return {
            "pg": round(pct / 100 * vals["n_gens"]),
            "qg": round(pct / 100 * vals["n_gens"]),
            "vm": round(pct / 100 * vals["n_buses"]),
            "f":  round(pct / 100 * vals["n_branches"]),
        }
    return (make_thresholds_for_case,)


@app.cell
def _(plt):
    def plot_pct_violations(df):
        labels = [f"{t} (≥{thr})" for t, thr in zip(df["type"], df["threshold"])]
        ax = df.plot.bar(
            x=None, y="percent", legend=False, figsize=(6,3), color="skyblue"
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=0)
        ax.set_ylabel("% of samples")
        ax.set_xlabel("Violation type (threshold)")
        ax.set_ylim(0, 100)
        ax.set_title("Exceedance percentages")

        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + 0.1, p.get_height() + 1))

        plt.tight_layout()
        plt.show()
    return (plot_pct_violations,)


if __name__ == "__main__":
    app.run()
