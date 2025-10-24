import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # PF$\Delta$ Dataset Validation Report
    **Purpose:** Verify that the dataset exhibits the claimed properties and characteristics.  
    **Dataset version:** v1.0  
    **Generated on:** Monday, October 13th, 2025
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    from notebooks.dataset_validation_utils import (
        all_samples_have_k_contingencies,
        add_ids_to_base,
        plot_grouped_outage_histograms,
        check_power_balance,
        build_outage_panels,
    )
    from core.datasets.pfdelta_dataset import PFDeltaDataset
    return (
        PFDeltaDataset,
        all_samples_have_k_contingencies,
        build_outage_panels,
        check_power_balance,
        mo,
    )


@app.cell
def _(PFDeltaDataset):
    # Load datasets to evaluate
    case_name = "case118"
    feasibility_type = "feasible"  # "feasible", "near infeasible"
    root_dir = "data"
    case_n = PFDeltaDataset(
        root_dir=root_dir,
        case_name=case_name,
        perturbation="n",
        task="analysis",
        n_samples=100,
        feasibility_type=feasibility_type,
    )
    case_n_1 = PFDeltaDataset(
        root_dir=root_dir,
        case_name=case_name,
        perturbation="n-1",
        task="analysis",
        n_samples=100,
        feasibility_type=feasibility_type,
    )

    case_n_2 = PFDeltaDataset(
        root_dir=root_dir,
        case_name=case_name,
        perturbation="n-2",
        task="analysis",
        n_samples=100,
        feasibility_type=feasibility_type,
    )
    return case_n, case_n_1, case_n_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Feasible samples""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### N-1, and N-2 topological perturbations""")
    return


@app.cell
def _(all_samples_have_k_contingencies, case_n, case_n_1):
    # Check if all N-1 samples have exactly 1 contingencies
    all_samples_have_k_contingencies(case_n[0], case_n_1, k=1)
    return


@app.cell
def _(all_samples_have_k_contingencies, case_n, case_n_2):
    # Check if all N-2 samples have exactly 2 contingencies
    all_samples_have_k_contingencies(case_n[0], case_n_2, k=2)
    return


@app.cell
def _(build_outage_panels, case_n, case_n_1, case_n_2):
    br_chart, gen_chart = build_outage_panels(
        case_n[0], case_n_1, case_n_2, normalize=False, title_prefix=" "
    )

    br_chart & gen_chart
    return


@app.cell
def _(mo):
    mo.md(r"""### Power Balance""")
    return


@app.cell
def _(case_n, check_power_balance):
    # Check power balance on N samples
    stats_n = check_power_balance(case_n)
    stats_n
    return


@app.cell
def _(case_n_1, check_power_balance):
    # Check power balance on N-1 samples
    stats_n_1 = check_power_balance(case_n_1)
    stats_n_1
    return


@app.cell
def _(case_n_2, check_power_balance):
    # Check power balance on N-2 samples
    stats_n_2 = check_power_balance(case_n_2)
    stats_n_2
    return


if __name__ == "__main__":
    app.run()
