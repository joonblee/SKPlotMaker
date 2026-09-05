from pathlib import Path

path = Path("qcd_bkg_estimation.py")
text = path.read_text()


def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one match, found {count}: {old[:100]!r}")
    text = text.replace(old, new, 1)


# Document the stabilised log-chi2 path.
replace_once(
    "      likelihood; chi2 and log-chi2 are available as cross-checks.\n"
    "The interval 9 < m(mumu) < 11 GeV is excluded from every QCD-MC fit objective\n",
    "      likelihood; chi2 and log-chi2 are available as cross-checks.  QCD-MC\n"
    "      log-chi2 fits use a weighted-likelihood prefit followed by a local\n"
    "      log-chi2 refinement to stabilise sparse Run-3 tails.\n"
    "The interval 9 < m(mumu) < 11 GeV is excluded from every QCD-MC fit objective\n",
)

# Let fit_one_model accept a preferred seed from the weighted-likelihood prefit.
replace_once(
    "    anchor: Optional[FitAnchor],\n"
    "    constraint: Optional[FitConstraint],\n"
    ") -> SelectedFit:\n"
    "    seeds = build_seed_list(model, selected_by_key, anchor, constraint)[:max_attempts]\n"
    "    candidates: List[FitCandidate] = []\n",
    "    anchor: Optional[FitAnchor],\n"
    "    constraint: Optional[FitConstraint],\n"
    "    preferred_seed: Optional[Tuple[float, ...]] = None,\n"
    ") -> SelectedFit:\n"
    "    seeds = build_seed_list(model, selected_by_key, anchor, constraint)\n"
    "    if preferred_seed is not None:\n"
    "        clipped_prefit_seed = clip_seed(preferred_seed, shape_bounds(model, constraint))\n"
    "        seeds = deduplicate_seeds([clipped_prefit_seed, *seeds])\n"
    "    seeds = seeds[:max_attempts]\n"
    "    candidates: List[FitCandidate] = []\n",
)

# Add a local parameter box around the stable weighted-likelihood prefit.
fit_all_marker = "\ndef fit_all_models(\n"
pos = text.index(fit_all_marker)
helper = r'''

def selected_shape_seed(selected: SelectedFit) -> Tuple[float, ...]:
    return tuple(
        float(selected.function.GetParameter(index))
        for index in range(1, selected.model.npar)
    )


def build_log_prefit_constraint(
    model: FitModelConfig,
    prefit: SelectedFit,
) -> FitConstraint:
    """Build broad local bounds around a weighted-likelihood QCD-MC prefit.

    The log-chi2 objective only contains positive bins and can develop remote
    local minima when the Run-3 tail is sparse.  The weighted-likelihood prefit
    locates the physically relevant basin using all non-negative weighted
    counts; log-chi2 then refines the shape inside a deliberately broad local
    neighbourhood.  This is not an SS-data constraint.
    """
    shape = selected_shape_seed(prefit)
    relative_half_width = {"n": 0.75, "k": 1.00, "w": 1.00}
    minimum_half_width = {"n": 1.00, "k": 0.10, "m0": 6.0, "w": 2.0}
    local_bounds = []
    for (name, global_low, global_high), value in zip(BASE_SHAPE_BOUNDS[model.key], shape):
        half = max(
            relative_half_width.get(name, 0.0) * abs(value),
            minimum_half_width[name],
        )
        low = max(global_low, value - half)
        high = min(global_high, value + half)
        if not high > low:
            low, high = global_low, global_high
        local_bounds.append((name, low, high))

    amplitude = max(abs(float(prefit.function.GetParameter(0))), 1e-20)
    return FitConstraint(
        source_label="weighted-likelihood prefit",
        anchor=FitAnchor(amplitude=amplitude, shape=shape),
        shape_bounds=tuple(local_bounds),
        amplitude_reference=amplitude,
        amplitude_bounds=(amplitude / 100.0, amplitude * 100.0),
    )


def selected_fit_rank(selected: SelectedFit, objective: str) -> Tuple[float, ...]:
    if objective == "log-chi2":
        log_metric = selected.metrics.log_chi2_ndf
        stat_metric = selected.metrics.stat_chi2_ndf
        return (
            0.0 if selected.accepted else 1.0,
            log_metric if math.isfinite(log_metric) else 1e99,
            stat_metric if math.isfinite(stat_metric) else 1e99,
        )
    return diagnostics_score(selected.diagnostics)
'''
text = text[:pos] + helper + text[pos:]

# Replace fit_all_models with the two-stage QCD log-chi2 strategy.
start = text.index("def fit_all_models(\n")
end = text.index("\n\n# =============================================================================\n# PLOTTING", start)
new_fit_all = r'''def fit_all_models(
    ROOT,
    density,
    fit_data: FitData,
    mode: ModeConfig,
    objective: str,
    args: argparse.Namespace,
    prefit_data: Optional[FitData] = None,
):
    colours = (
        ROOT.kBlue + 1, ROOT.kRed + 1, ROOT.kGreen + 2,
        ROOT.kOrange + 7, ROOT.kCyan + 2, ROOT.kViolet + 1,
    )
    line_styles = (1, 1, 1, 1, 1, 1)
    model_by_key = {m.key: m for m in FIT_MODELS}

    dynamic_catalog = load_dynamic_anchor_catalog(args)
    resolved_anchors = {
        model.key: get_qcd_anchor(
            mode,
            args.year,
            model,
            dynamic_catalog,
            args.prefer_full_run_anchor,
        )
        for model in FIT_MODELS
    }
    anchors = {
        key: (resolved.anchor if resolved is not None else None)
        for key, resolved in resolved_anchors.items()
    }

    if mode.key == QCD_MODE.key and args.initial_values_only:
        constraints = {model.key: None for model in FIT_MODELS}
        for model in FIT_MODELS:
            print_qcd_initial_only(
                ROOT, density, mode, args.year, model,
                resolved_anchors[model.key],
            )
    else:
        constraints = {
            model.key: build_qcd_constraint(
                ROOT, density, mode, args.year, model,
                resolved_anchors[model.key],
            )
            for model in FIT_MODELS
        }

    internal_order = (
        "power_erf", "power_logistic", "exp_erf", "exp_logistic",
        "power_exp_erf", "power_exp_logistic",
    )

    # QCD log-chi2 is sensitive to sparse positive-only Run-3 tails.  First
    # locate the correct global basin with the weighted-likelihood objective,
    # then use that solution as the first seed and (unless the user explicitly
    # requested SS-anchor hard constraints) as a broad local parameter box.
    prefit_by_key: Dict[str, SelectedFit] = {}
    prefit_seeds: Dict[str, Tuple[float, ...]] = {}
    effective_constraints = dict(constraints)
    if mode.key == QCD_MODE.key and objective == "log-chi2":
        if prefit_data is None:
            raise RuntimeError("QCD log-chi2 requires a weighted-likelihood prefit dataset.")
        prefit_attempts = min(args.fit_max_attempts, 12)
        print(
            "[LOG-CHI2 PREFIT] Running weighted-likelihood prefit before "
            f"log-chi2 refinement (max attempts/model={prefit_attempts})."
        )
        for key in internal_order:
            prefit_by_key[key] = fit_one_model(
                ROOT, density, prefit_data, model_by_key[key], mode,
                "weighted-likelihood", prefit_attempts,
                args.log_relative_error_floor, False,
                prefit_by_key, anchors[key], constraints[key],
            )
            prefit_seeds[key] = selected_shape_seed(prefit_by_key[key])
            if constraints[key] is None:
                effective_constraints[key] = build_log_prefit_constraint(
                    model_by_key[key], prefit_by_key[key]
                )
                bounds_text = ", ".join(
                    f"{name}=[{low:g},{high:g}]"
                    for name, low, high in effective_constraints[key].shape_bounds
                )
                print(
                    f"[LOG-CHI2 PREFIT] model={model_by_key[key].label.replace('#times','x')} "
                    f"seed={prefit_seeds[key]} localBounds: {bounds_text}"
                )

    selected_by_key: Dict[str, SelectedFit] = {}
    for key in internal_order:
        selected_by_key[key] = fit_one_model(
            ROOT, density, fit_data, model_by_key[key], mode, objective,
            args.fit_max_attempts, args.log_relative_error_floor,
            args.fit_attempt_details, selected_by_key, anchors[key],
            effective_constraints[key], prefit_seeds.get(key),
        )

    # Recovery pass.  In log-chi2 mode, an accepted Minuit minimum can still be
    # clearly pathological.  A very large recomputed log-chi2/ndf therefore
    # triggers one SS-anchor-constrained retry; the retry is retained only when
    # it actually improves the requested objective.
    for key in internal_order:
        current = selected_by_key[key]
        pathological_log = (
            mode.key == QCD_MODE.key
            and objective == "log-chi2"
            and (
                not math.isfinite(current.metrics.log_chi2_ndf)
                or current.metrics.log_chi2_ndf > 20.0
            )
        )
        if current.accepted and not pathological_log:
            continue

        retry_constraint = effective_constraints[key]
        recovery_reason = "invalid fit"
        if pathological_log:
            recovery_reason = f"logChi2/ndf={current.metrics.log_chi2_ndf:.6g}"
            if constraints[key] is None:
                retry_constraint = build_qcd_constraint(
                    ROOT, density, mode, args.year, model_by_key[key],
                    resolved_anchors[key],
                )
        print(
            f"[FIT RECOVERY] model={model_by_key[key].label.replace('#times','x')} "
            f"reason={recovery_reason}"
        )
        recovered = fit_one_model(
            ROOT, density, fit_data, model_by_key[key], mode, objective,
            args.fit_max_attempts, args.log_relative_error_floor,
            args.fit_attempt_details, selected_by_key, anchors[key],
            retry_constraint, prefit_seeds.get(key),
        )
        if selected_fit_rank(recovered, objective) < selected_fit_rank(current, objective):
            selected_by_key[key] = recovered

    selected = []
    for index, model in enumerate(FIT_MODELS):
        outcome = selected_by_key[model.key]
        outcome.function.SetName(f"fit_{index}")
        outcome.function.SetLineColor(colours[index])
        outcome.function.SetLineWidth(2)
        outcome.function.SetLineStyle(line_styles[index])
        selected.append(outcome)
    return selected, colours, line_styles
'''
text = text[:start] + new_fit_all + text[end:]

# Plot cosmetics requested by the analysis.
replace_once(
    '    latex.DrawLatex(0.955, 0.935, cms_lumi_label(period))\n',
    '    latex.DrawLatex(0.935, 0.935, cms_lumi_label(period))\n',
)
replace_once('    upper.SetLeftMargin(0.1)\n', '    upper.SetLeftMargin(0.12)\n')
replace_once('    lower.SetLeftMargin(0.1)\n', '    lower.SetLeftMargin(0.12)\n')
replace_once(
    '    density.GetYaxis().SetTitle("Events (log)")\n',
    '    density.GetYaxis().SetTitle("Events/GeV (log)")\n',
)
replace_once(
    '    density.GetYaxis().SetTitleOffset(1.0)\n',
    '    density.GetYaxis().SetTitleOffset(1.18)\n',
)

# Build the weighted-likelihood prefit data only for QCD log-chi2.
replace_once(
    "        fit_data = build_fit_data(\n"
    "            ROOT, prepared, mode, objective, args.log_relative_error_floor\n"
    "        )\n"
    "        selected, colours, styles = fit_all_models(\n"
    "            ROOT, density, fit_data, mode, objective, args\n"
    "        )\n",
    "        fit_data = build_fit_data(\n"
    "            ROOT, prepared, mode, objective, args.log_relative_error_floor\n"
    "        )\n"
    "        prefit_data = None\n"
    "        if mode.key == QCD_MODE.key and objective == \"log-chi2\":\n"
    "            prefit_data = build_fit_data(\n"
    "                ROOT, prepared, mode, \"weighted-likelihood\",\n"
    "                args.log_relative_error_floor,\n"
    "            )\n"
    "            print(\"[INFO] log-chi2 stabilisation = weighted-likelihood prefit + local refinement\")\n"
    "        selected, colours, styles = fit_all_models(\n"
    "            ROOT, density, fit_data, mode, objective, args, prefit_data\n"
    "        )\n",
)

path.write_text(text)
