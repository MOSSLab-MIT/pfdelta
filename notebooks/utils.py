import pandas as pd


def check_violations(dataset):
    # Arrays to keep track of violations
    violations_results = {"pg": {}, "qg": {}, "vm": {}, "f": {}}

    # Parse limits from case file

    i = 0
    for data in dataset:
        # Check for generator violations
        pg, qg = data["gen"]["generation"].T
        pmin, pmax, qmin, qmax = data["gen"]["limits"].T

        pg_violations_mask = (pg < pmin) | (pg > pmax)
        qg_violations_mask = (qg < qmin) | (qg > qmax)

        violations_results["pg"][i] = pg_violations_mask.sum().item()
        violations_results["qg"][i] = qg_violations_mask.sum().item()
        # TODO: what if a gnerator is off? check ordering of generators as well (is it consistent with mpc?)

        # Check for voltage violations
        _, vm = data["bus"]["bus_voltages"].T
        vmin, vmax = data["bus"]["voltage_limits"].T
        v_violations_mask = (vm < vmin) | (vm > vmax)

        violations_results["vm"][i] = v_violations_mask.sum().item()

        # Check for line flow violations
        pf, qf, pt, qt = data["bus", "branch", "bus"].edge_label.T
        fmax = data["bus", "branch", "bus"].edge_limits.T
        st, sf = (pt**2 + qt**2).sqrt(), (pf**2 + qf**2).sqrt()
        f_violations_mask = (sf > fmax) | (st > fmax)

        violations_results["f"][i] = f_violations_mask.sum().item()

        i += 1

    return violations_results


def compute_pct_violations(violations_df, thresholds):
    results = []
    for vtype, trh in thresholds.items():
        counts = violations_df[vtype].to_numpy()
        pct = (counts >= trh).sum() / len(counts) * 100.0
        results.append({"type": vtype, "threshold": trh, "percent": pct})

    return pd.DataFrame(results)

