"""ROOT-free common QCD MC transport and cross-era statistical propagation.

The fit is GLS in log(D), using first-order Sumw2 errors. It tests statistical
compatibility, not equality or coverage of detector/generator modelling effects.
"""

import hashlib
import json
import math

COMMON_STAT_SCHEMA = "NPS26009_QCDStat_v3"
COMMON_STAT_TREATMENT = "local_nf_fit_bound_plus_shared_mc_transport"
GROUP_ERAS = {
    "Run2": ("2016preVFP", "2016postVFP", "2017", "2018"),
    "Run3": ("2022", "2022EE", "2023", "2023BPix"),
}
MC_KEYS = ("mc_os_low", "mc_ss_low", "mc_os_high", "mc_ss_high")


def transfer_group(era):
    for group, eras in GROUP_ERAS.items():
        if era in eras:
            return group
    raise ValueError(f"No QCD transfer group for {era}")


def mc_double_ratio(primitive):
    for key in MC_KEYS:
        y, v = primitive[key]["value"], primitive[key]["variance"]
        if not (math.isfinite(y) and y > 0 and math.isfinite(v) and v > 0):
            raise ValueError(f"Invalid QCD MC yield/Sumw2 for {key}: {y}, {v}")
    p = primitive
    d = p["mc_os_high"]["value"] * p["mc_ss_low"]["value"] / (
        p["mc_ss_high"]["value"] * p["mc_os_low"]["value"])
    log_variance = math.fsum(p[key]["variance"] / p[key]["value"] ** 2 for key in MC_KEYS)
    return d, log_variance


def _solve_spd(matrix, vector):
    """Cholesky solve; reject asymmetric/singular covariance without regularising."""
    n = len(vector)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("QCD log-ratio covariance dimensions disagree with eras")
    if any(not math.isfinite(x) for row in matrix for x in row):
        raise ValueError("Non-finite QCD log-ratio covariance")
    if any(not math.isclose(matrix[i][j], matrix[j][i], rel_tol=1e-10, abs_tol=1e-15)
           for i in range(n) for j in range(n)):
        raise ValueError("Asymmetric QCD log-ratio covariance")
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            x = matrix[i][j] - math.fsum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if x <= 0:
                    raise ValueError("QCD log-ratio covariance is not positive definite")
                lower[i][j] = math.sqrt(x)
            else:
                lower[i][j] = x / lower[j][j]
    forward = []
    for i in range(n):
        forward.append((vector[i] - math.fsum(lower[i][j] * forward[j] for j in range(i))) / lower[i][i])
    result = [0.0] * n
    for i in reversed(range(n)):
        result[i] = (forward[i] - math.fsum(lower[j][i] * result[j] for j in range(i + 1, n))) / lower[i][i]
    return result


def _constant_fit(values, covariance):
    precision = _solve_spd(covariance, [1.0] * len(values))
    total = math.fsum(precision)
    if total <= 0:
        raise ValueError("Invalid QCD common-fit precision")
    weights = [x / total for x in precision]
    return math.fsum(w * x for w, x in zip(weights, values)), 1.0 / total, weights


def chi2_survival(chi2, ndf):
    """Chi-square survival for integer degrees of freedom; no SciPy dependency."""
    if ndf <= 0 or chi2 < 0 or not math.isfinite(chi2):
        raise ValueError("Invalid QCD compatibility chi-square")
    if chi2 == 0:
        return 1.0
    x = chi2 / 2.0
    a = 0.5 if ndf % 2 else 1.0
    result = math.erfc(math.sqrt(x)) if ndf % 2 else math.exp(-x)
    while a < ndf / 2.0:
        result += math.exp(a * math.log(x) - x - math.lgamma(a + 1.0))
        a += 1.0
    return min(1.0, max(0.0, result))


def fit_common_double_ratio(records, group, windows, log_covariance=None, pmin=0.05):
    eras = GROUP_ERAS[group]
    if set(records) != set(eras):
        raise ValueError(f"Common {group} QCD transport requires all eras: {', '.join(eras)}")
    if not 0 < pmin < 1:
        raise ValueError("QCD compatibility p-value threshold must be between 0 and 1")
    measurements = [mc_double_ratio(records[era]["primitive"]) for era in eras]
    values = [math.log(d) for d, _ in measurements]
    covariance = (log_covariance if log_covariance is not None else
                  [[v if i == j else 0.0 for j in range(len(eras))]
                   for i, (_, v) in enumerate(measurements)])
    for i, (_, variance) in enumerate(measurements):
        if (len(covariance) != len(eras) or len(covariance[i]) != len(eras)
                or not math.isclose(covariance[i][i], variance, rel_tol=1e-8, abs_tol=1e-15)):
            raise ValueError("QCD statistical log covariance diagonal must match primitive Sumw2 propagation")
    mean, variance, weights = _constant_fit(values, covariance)
    residuals = [x - mean for x in values]
    chi2 = max(0.0, math.fsum(x * y for x, y in zip(residuals, _solve_spd(covariance, residuals))))
    pvalue = chi2_survival(chi2, len(eras) - 1)
    rows = {}
    for i, era in enumerate(eras):
        keep = [j for j in range(len(eras)) if j != i]
        cov_loo = [[covariance[j][k] for k in keep] for j in keep]
        loo_mean, loo_variance, loo_weights = _constant_fit([values[j] for j in keep], cov_loo)
        cross = math.fsum(w * covariance[i][j] for w, j in zip(loo_weights, keep))
        prediction_variance = covariance[i][i] + loo_variance - 2.0 * cross
        if prediction_variance <= 0:
            raise ValueError("Invalid QCD leave-one-out prediction variance")
        pull = (values[i] - loo_mean) / math.sqrt(prediction_variance)
        d, v = measurements[i]
        rows[era] = dict(double_ratio=d, variance=d * d * v, log_variance=v,
                         error_low=d * (1.0 - math.exp(-math.sqrt(v))),
                         error_high=d * math.expm1(math.sqrt(v)),
                         fit_weight=weights[i], leave_one_out_double_ratio=math.exp(loo_mean),
                         leave_one_out_pull=pull, within_1sigma=abs(pull) <= 1.0,
                         within_2sigma=abs(pull) <= 2.0,
                         effective_counts={key: records[era]["primitive"][key]["value"] ** 2 /
                                           records[era]["primitive"][key]["variance"] for key in MC_KEYS})
    d = math.exp(mean)
    identity_inputs = dict(group=group, eras=list(eras), windows=windows, records=records,
                           log_covariance=covariance, estimator="GLS_log_double_ratio_v1")
    identity = hashlib.sha256(json.dumps(identity_inputs, sort_keys=True, allow_nan=False,
                                        separators=(",", ":")).encode()).hexdigest()
    return dict(schema="NPS26009_QCDCommonTransfer_v1", id=identity, **identity_inputs,
                double_ratio=d, variance=d * d * variance, log_mean=mean, log_variance=variance,
                error_low=d * (1.0 - math.exp(-math.sqrt(variance))),
                error_high=d * math.expm1(math.sqrt(variance)), chi2=chi2,
                ndf=len(eras) - 1, pvalue=pvalue, pmin=pmin,
                compatible=pvalue >= pmin, rows=rows,
                assumptions=["First-order Gaussian approximation for log(D) from weighted MC Sumw2.",
                             "Statistical covariance only; detector/generator modelling is not tested.",
                             "Independent per-era MC statistics unless a log covariance is supplied.",
                             "Compatibility does not establish equality, especially with sparse MC."])


def validate_common_transfer(transfer, era=None):
    common = transfer["common_transfer"]
    if common.get("schema") != "NPS26009_QCDCommonTransfer_v1":
        raise ValueError("Unsupported common QCD transfer metadata")
    group = common["group"]
    if era is not None and transfer_group(era) != group:
        raise ValueError("Common QCD transport belongs to a different Run")
    rebuilt = fit_common_double_ratio(common["records"], group, transfer["windows"],
                                     common["log_covariance"], common["pmin"])
    if common["id"] != rebuilt["id"] or common["eras"] != rebuilt["eras"]:
        raise ValueError("Common QCD transfer inputs/windows/identity disagree")
    for key in ("double_ratio", "variance", "log_mean", "log_variance", "chi2", "pvalue"):
        if not math.isclose(common[key], rebuilt[key], rel_tol=1e-10, abs_tol=1e-14):
            raise ValueError(f"Common QCD transfer {key} disagrees with its inputs")
    if (common["compatible"] != rebuilt["compatible"] or common["ndf"] != rebuilt["ndf"]
            or common["windows"] != transfer["windows"] or common["rows"] != rebuilt["rows"]
            or common["estimator"] != rebuilt["estimator"]):
        raise ValueError("Common QCD compatibility result/windows disagree with its inputs")
    if era is not None:
        for key in MC_KEYS:
            if common["records"][era]["primitive"][key] != transfer["primitive"][key]:
                raise ValueError(f"{era}: common QCD fit and local MC primitive {key} disagree")
    for key, expected in (("mc_double_ratio", common["double_ratio"]),
                          ("double_ratio_variance", common["variance"])):
        if not math.isclose(transfer[key], expected, rel_tol=1e-10, abs_tol=1e-15):
            raise ValueError(f"Applied QCD {key} disagrees with the common fit")
    return common


def qcd_stat_components(transfer, low_gradient, mc_gradient, fit_sigma):
    local_nf = abs(low_gradient) * math.sqrt(transfer["low_variance"])
    mc_sigma = abs(mc_gradient) * math.sqrt(transfer["double_ratio_variance"])
    nf_sigma = math.hypot(local_nf, mc_sigma)
    common = transfer.get("common_transfer") if transfer.get("method") == "run-common" else None
    local_bound = local_nf + fit_sigma if common else nf_sigma + fit_sigma
    total = math.hypot(local_bound, mc_sigma) if common else local_bound
    return dict(sigma_nf_stat=nf_sigma, sigma_nf_low_stat=local_nf,
                sigma_mc_transfer_stat=mc_sigma, sigma_local_stat_bound=local_bound,
                sigma_stat_bound=total,
                common_transfer_group=common["group"] if common else None,
                common_transfer_id=common["id"] if common else None)


def combine_qcd_stat_components(results):
    """Era-local conservative bounds in quadrature; common MC shifts linearly."""
    groups, signatures = {}, {}
    local_variance = 0.0
    for era, result in results.items():
        group = transfer_group(era)
        identity = result["common_transfer_id"]
        if identity and result["common_transfer_group"] != group:
            raise ValueError(f"{era}: common QCD statistical component belongs to a different Run")
        if group in signatures and signatures[group] != identity:
            raise ValueError(f"Mixed/stale common QCD transport in {group}; regenerate all affected era templates")
        signatures[group] = identity
        local_variance += result["sigma_local_stat_bound"] ** 2
        if identity:
            groups[group] = groups.get(group, 0.0) + result["sigma_mc_transfer_stat"]
    return math.sqrt(local_variance + math.fsum(x * x for x in groups.values()))
