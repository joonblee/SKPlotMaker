#!/usr/bin/env bash

# NIsoMuon Run-2/Run-3 ROOT-file merger
# Updated for the NPS-26-009 SKNano production layout (2026-08-25).
#
# Usage:
#   source hadd.sh ERA_SELECTION [COLLECTION]
#   bash   hadd.sh ERA_SELECTION [COLLECTION]
#
# ERA_SELECTION:
#   2016preVFP, 2016postVFP, 2017, 2018,
#   2022, 2022EE, 2023, 2023BPix,
#   Run2, Run3, Run2+3, all
#
# COLLECTION:
#   nominal           : <BASE>/<era>
#   RunSyst           : <BASE>/RunSyst/<era>
#   RunXSecSyst       : <BASE>/RunXSecSyst/<era>
#   MuonIDEfficiency  : <BASE>/MuonIDEfficiency/<era>
#   TriggerEfficiency : <BASE>/TriggerEfficiency/<era>
#   all               : all five
#
# Current production policy:
#   nominal           : data + all nominal backgrounds + signal
#   RunSyst           : tt/ST/Others + signal, OS BJet histograms only
#   RunXSecSyst       : tt/ST/Others only, OS BJet histograms only
#   MuonIDEfficiency  : merge the available efficiency skim files only; no signal
#   TriggerEfficiency : merge the available efficiency skim files only; no signal
#
# Important details:
#   * Run-3 TTLL_powheg_ext1 is included in tt where it exists.
#   * A separate NIsoMuon_tW.root is made so the tW-only normalisation
#     uncertainty can be propagated without assigning it to all single top.
#   * pT-binned Z' samples are merged into NIsoMuon_Zp_M-<mass>.root.
#   * Mixed summary files (AllMC/QCDTop) are made for nominal and efficiency collections.

print_help() {
    echo "NIsoMuon Run-2/Run-3 ROOT-file merger"
    echo
    echo "Fixed base directory:"
    echo "  /data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon"
    echo
    echo "Usage:"
    echo "  source hadd.sh ERA_SELECTION [COLLECTION]"
    echo "  bash   hadd.sh ERA_SELECTION [COLLECTION]"
    echo
    echo "ERA_SELECTION:"
    echo "  2016preVFP, 2016postVFP, 2017, 2018,"
    echo "  2022, 2022EE, 2023, 2023BPix,"
    echo "  Run2, Run3, Run2+3, all"
    echo
    echo "COLLECTION (default: nominal):"
    echo "  nominal, RunSyst, RunXSecSyst,"
    echo "  MuonIDEfficiency, TriggerEfficiency, all"
}

run_hadd() {
    local output="$1"
    shift

    local inputs=()
    local file
    for file in "$@"; do
        if [[ -f "$file" ]]; then
            inputs+=("$file")
        fi
    done

    if (( ${#inputs[@]} == 0 )); then
        echo "[skip] no input files for $output"
        return 0
    fi

    mkdir -p "$(dirname "$output")"
    rm -f "$output"

    echo
    echo "[hadd] $output"
    for file in "${inputs[@]}"; do
        echo "       $file"
    done

    hadd -f "$output" "${inputs[@]}"
    local status=$?
    if (( status != 0 )); then
        echo "[ERROR] hadd failed: $output"
        return "$status"
    fi

    echo "[done] $output"
    return 0
}

is_run2() {
    case "$1" in
        2016preVFP|2016postVFP|2017|2018) return 0 ;;
        *) return 1 ;;
    esac
}

is_efficiency_collection() {
    case "$1" in
        MuonIDEfficiency|TriggerEfficiency) return 0 ;;
        *) return 1 ;;
    esac
}

is_nominal_like_collection() {
    [[ "$1" == "nominal" ]] || is_efficiency_collection "$1"
}

collection_has_signal() {
    [[ "$1" == "nominal" || "$1" == "RunSyst" ]]
}

get_dir() {
    local era="$1"
    local collection="$2"
    if [[ "$collection" == "nominal" ]]; then
        DIR="$BASE_DIR/$era"
    else
        DIR="$BASE_DIR/$collection/$era"
    fi
}

merge_data() {
    local era="$1"
    local dir="$2"
    local inputs=()

    if is_run2 "$era"; then
        inputs=( "$dir"/Skim_NIsoMuon_SingleMuon_*.root )
    elif [[ "$era" == "2022" || "$era" == "2022EE" ]]; then
        inputs=( "$dir"/Skim_NIsoMuon_Muon_*.root )
    else
        inputs=(
            "$dir"/Skim_NIsoMuon_Muon0_*.root
            "$dir"/Skim_NIsoMuon_Muon1_*.root
        )
    fi

    run_hadd "$dir/data.root" "${inputs[@]}"
}

merge_qcd() {
    local dir="$1"
    local inputs=( "$dir"/Skim_NIsoMuon_QCD_Pt-*_MuEnriched.root )
    run_hadd "$dir/NIsoMuon_QCD_Inclusive.root" "${inputs[@]}"
}

merge_dy() {
    local dir="$1"

    local amc_inputs=(
        "$dir/Skim_NIsoMuon_DYJets.root"
        "$dir/Skim_NIsoMuon_DYJets10to50.root"
    )
    local mg_inputs=(
        "$dir/Skim_NIsoMuon_DYJets_MG.root"
        "$dir/Skim_NIsoMuon_DYJets10to50_MG.root"
    )

    run_hadd "$dir/NIsoMuon_DYJets_Inclusive.root" "${amc_inputs[@]}"
    run_hadd "$dir/NIsoMuon_DYJets_MG_Inclusive.root" "${mg_inputs[@]}"
}

merge_st() {
    local era="$1"
    local dir="$2"
    local inputs=()

    if is_run2 "$era"; then
        inputs=( "$dir"/Skim_NIsoMuon_SingleTop_*.root )
    else
        inputs=( "$dir"/Skim_NIsoMuon_ST_*.root )
    fi

    run_hadd "$dir/NIsoMuon_ST.root" "${inputs[@]}"
}

merge_tw() {
    local era="$1"
    local dir="$2"
    local inputs=()

    if is_run2 "$era"; then
        inputs=(
            "$dir/Skim_NIsoMuon_SingleTop_tW_top_NoFullyHad.root"
            "$dir/Skim_NIsoMuon_SingleTop_tW_antitop_NoFullyHad.root"
        )
    else
        inputs=(
            "$dir/Skim_NIsoMuon_ST_tW_top_Semilep.root"
            "$dir/Skim_NIsoMuon_ST_tW_antitop_Semilep.root"
            "$dir/Skim_NIsoMuon_ST_tW_top_Lep.root"
            "$dir/Skim_NIsoMuon_ST_tW_antitop_Lep.root"
        )
    fi

    run_hadd "$dir/NIsoMuon_tW.root" "${inputs[@]}"
}

merge_ttbar() {
    local era="$1"
    local dir="$2"
    local inputs=(
        "$dir/Skim_NIsoMuon_TTLL_powheg.root"
        "$dir/Skim_NIsoMuon_TTLJ_powheg.root"
        "$dir/Skim_NIsoMuon_TTJJ_powheg.root"
    )

    # Run-3 uses TTLL_powheg_ext1 as a nominal extension in eras where it exists.
    if ! is_run2 "$era"; then
        inputs+=( "$dir/Skim_NIsoMuon_TTLL_powheg_ext1.root" )
    fi

    run_hadd "$dir/NIsoMuon_tt.root" "${inputs[@]}"
}

merge_top() {
    local dir="$1"
    local inputs=(
        "$dir/NIsoMuon_ST.root"
        "$dir/NIsoMuon_tt.root"
    )
    run_hadd "$dir/NIsoMuon_Top.root" "${inputs[@]}"
}

merge_others() {
    local era="$1"
    local dir="$2"
    local inputs=()

    if is_run2 "$era"; then
        local ttz_file="$dir/Skim_NIsoMuon_TTZToLLNuNu.root"
        if [[ ! -f "$ttz_file" && -f "$dir/Skim_NIsoMuon_ttZToLLNuNu.root" ]]; then
            ttz_file="$dir/Skim_NIsoMuon_ttZToLLNuNu.root"
        fi
        inputs=(
            "$dir/Skim_NIsoMuon_WJets_MG.root"
            "$dir/Skim_NIsoMuon_TTG.root"
            "$ttz_file"
        )
    else
        inputs=(
            "$dir/Skim_NIsoMuon_WJets_MG.root"
            "$dir/Skim_NIsoMuon_TTZ_NoFullyHad.root"
            "$dir/Skim_NIsoMuon_TTG_PTG10to100.root"
            "$dir/Skim_NIsoMuon_TTG_PTG100to200.root"
            "$dir/Skim_NIsoMuon_TTG_PTG200toInf.root"
        )
    fi

    run_hadd "$dir/NIsoMuon_Others.root" "${inputs[@]}"
}

merge_signals() {
    local dir="$1"
    local files=( "$dir"/Skim_NIsoMuon_Zp_M-*.root )
    local masses=()
    local file base mass
    declare -A seen_mass=()

    for file in "${files[@]}"; do
        base="$(basename "$file")"
        if [[ "$base" =~ ^Skim_NIsoMuon_Zp_M-([0-9]+([p.][0-9]+)?)(_Pt-|\.root) ]]; then
            mass="${BASH_REMATCH[1]}"
            if [[ -z "${seen_mass[$mass]+x}" ]]; then
                masses+=("$mass")
                seen_mass[$mass]=1
            fi
        fi
    done

    if (( ${#masses[@]} == 0 )); then
        echo "[skip] no signal input files under $dir"
        return 0
    fi

    # Numerical sort keeps 12,15,...,70 in a predictable order.
    mapfile -t masses < <(printf '%s\n' "${masses[@]}" | sort -g)

    for mass in "${masses[@]}"; do
        local inputs=(
            "$dir"/Skim_NIsoMuon_Zp_M-"$mass"_Pt-*_hw7.root
            "$dir"/Skim_NIsoMuon_Zp_M-"$mass"_Pt-*.root
            "$dir"/Skim_NIsoMuon_Zp_M-"$mass".root
        )

        # Remove duplicates because the generic _Pt-* glob also matches _Pt-*_hw7.
        local unique_inputs=()
        declare -A seen_file=()
        local input
        for input in "${inputs[@]}"; do
            [[ -f "$input" ]] || continue
            if [[ -z "${seen_file[$input]+x}" ]]; then
                unique_inputs+=("$input")
                seen_file[$input]=1
            fi
        done

        run_hadd "$dir/NIsoMuon_Zp_M-$mass.root" "${unique_inputs[@]}" || return $?
    done

    return 0
}

merge_summary() {
    local dir="$1"

    local allmc_inputs=(
        "$dir/NIsoMuon_DYJets_Inclusive.root"
        "$dir/NIsoMuon_Top.root"
        "$dir/NIsoMuon_QCD_Inclusive.root"
        "$dir/NIsoMuon_Others.root"
    )
    local qcdtop_inputs=(
        "$dir/NIsoMuon_Top.root"
        "$dir/NIsoMuon_QCD_Inclusive.root"
    )

    run_hadd "$dir/NIsoMuon_AllMC.root" "${allmc_inputs[@]}" || return $?
    run_hadd "$dir/NIsoMuon_QCDTop.root" "${qcdtop_inputs[@]}" || return $?
    return 0
}

run_one() {
    local era="$1"
    local collection="$2"

    get_dir "$era" "$collection"
    local dir="$DIR"

    echo
    echo "================================================================"
    echo "[era]        $era"
    echo "[collection] $collection"
    echo "[dir]        $dir"
    echo "================================================================"

    if [[ ! -d "$dir" ]]; then
        echo "[skip] directory does not exist: $dir"
        return 0
    fi

    if is_nominal_like_collection "$collection"; then
        merge_data "$era" "$dir" || return $?
        merge_qcd "$dir" || return $?
        merge_dy "$dir" || return $?
    fi

    # Systematic and efficiency productions are merged from whatever restricted
    # process set is present in the directory.
    merge_st "$era" "$dir" || return $?
    merge_tw "$era" "$dir" || return $?
    merge_ttbar "$era" "$dir" || return $?
    merge_top "$dir" || return $?
    merge_others "$era" "$dir" || return $?

    # Signal is produced only for the nominal and RunSyst productions.
    # Efficiency collections intentionally contain no signal skim files.
    if collection_has_signal "$collection"; then
        merge_signals "$dir" || return $?
    fi

    # Efficiency collections contain the same background skim families as the
    # nominal production, so build the standard summary files there as well.
    if is_nominal_like_collection "$collection"; then
        merge_summary "$dir" || return $?
    fi

    return 0
}

get_period_dir() {
    local period="$1"
    local collection="$2"
    if [[ "$collection" == "nominal" ]]; then
        PERIOD_DIR="$BASE_DIR/$period"
    else
        PERIOD_DIR="$BASE_DIR/$collection/$period"
    fi
}

get_era_dir() {
    local era="$1"
    local collection="$2"
    if [[ "$collection" == "nominal" ]]; then
        ERA_DIR="$BASE_DIR/$era"
    else
        ERA_DIR="$BASE_DIR/$collection/$era"
    fi
}

merge_period_file() {
    local period="$1"
    local collection="$2"
    local filename="$3"
    shift 3

    get_period_dir "$period" "$collection"
    local output="$PERIOD_DIR/$filename"
    local inputs=()
    local era

    for era in "$@"; do
        get_era_dir "$era" "$collection"
        [[ -f "$ERA_DIR/$filename" ]] && inputs+=( "$ERA_DIR/$filename" )
    done

    run_hadd "$output" "${inputs[@]}"
}

merge_period_signals() {
    local period="$1"
    local collection="$2"
    shift 2
    local eras=( "$@" )
    local masses=()
    local era file base mass
    declare -A seen_mass=()

    for era in "${eras[@]}"; do
        get_era_dir "$era" "$collection"
        for file in "$ERA_DIR"/NIsoMuon_Zp_M-*.root; do
            [[ -f "$file" ]] || continue
            base="$(basename "$file")"
            if [[ "$base" =~ ^NIsoMuon_Zp_M-([0-9]+([p.][0-9]+)?)\.root$ ]]; then
                mass="${BASH_REMATCH[1]}"
                if [[ -z "${seen_mass[$mass]+x}" ]]; then
                    masses+=("$mass")
                    seen_mass[$mass]=1
                fi
            fi
        done
    done

    mapfile -t masses < <(printf '%s\n' "${masses[@]}" | sed '/^$/d' | sort -g)
    for mass in "${masses[@]}"; do
        merge_period_file "$period" "$collection" "NIsoMuon_Zp_M-$mass.root" "${eras[@]}" || return $?
    done
    return 0
}

merge_period() {
    local period="$1"
    local collection="$2"
    shift 2
    local eras=( "$@" )

    get_period_dir "$period" "$collection"

    echo
    echo "################################################################"
    echo "[period merge] $period"
    echo "[collection]   $collection"
    echo "[output dir]   $PERIOD_DIR"
    echo "################################################################"

    if is_nominal_like_collection "$collection"; then
        merge_period_file "$period" "$collection" "data.root" "${eras[@]}" || return $?
        merge_period_file "$period" "$collection" "NIsoMuon_QCD_Inclusive.root" "${eras[@]}" || return $?
        merge_period_file "$period" "$collection" "NIsoMuon_DYJets_Inclusive.root" "${eras[@]}" || return $?
        merge_period_file "$period" "$collection" "NIsoMuon_DYJets_MG_Inclusive.root" "${eras[@]}" || return $?
    fi

    merge_period_file "$period" "$collection" "NIsoMuon_ST.root" "${eras[@]}" || return $?
    merge_period_file "$period" "$collection" "NIsoMuon_tW.root" "${eras[@]}" || return $?
    merge_period_file "$period" "$collection" "NIsoMuon_tt.root" "${eras[@]}" || return $?
    merge_period_file "$period" "$collection" "NIsoMuon_Top.root" "${eras[@]}" || return $?
    merge_period_file "$period" "$collection" "NIsoMuon_Others.root" "${eras[@]}" || return $?

    if collection_has_signal "$collection"; then
        merge_period_signals "$period" "$collection" "${eras[@]}" || return $?
    fi

    if is_nominal_like_collection "$collection"; then
        merge_period_file "$period" "$collection" "NIsoMuon_AllMC.root" "${eras[@]}" || return $?
        merge_period_file "$period" "$collection" "NIsoMuon_QCDTop.root" "${eras[@]}" || return $?
    fi

    return 0
}

hadd_main() {
    if (( $# == 0 )); then
        print_help
        return 0
    fi

    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        print_help
        return 0
    fi

    if (( $# > 2 )); then
        echo "[ERROR] Too many arguments."
        print_help
        return 2
    fi

    BASE_DIR="/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon"

    local era_selection="$1"
    local collection_selection="${2:-nominal}"

    local run2_eras=(2016preVFP 2016postVFP 2017 2018)
    local run3_eras=(2022 2022EE 2023 2023BPix)
    local eras=()
    local collections=()
    local make_run2_summary=0
    local make_run3_summary=0

    case "$era_selection" in
        Run2|run2)
            eras=("${run2_eras[@]}")
            make_run2_summary=1
            ;;
        Run3|run3)
            eras=("${run3_eras[@]}")
            make_run3_summary=1
            ;;
        Run2+3|run2+3|Run23|run23|all)
            eras=("${run2_eras[@]}" "${run3_eras[@]}")
            make_run2_summary=1
            make_run3_summary=1
            ;;
        2016preVFP|2016postVFP|2017|2018|2022|2022EE|2023|2023BPix)
            eras=("$era_selection")
            ;;
        *)
            echo "[ERROR] Unknown ERA_SELECTION: $era_selection"
            print_help
            return 2
            ;;
    esac

    case "$collection_selection" in
        nominal|Nominal|"") collections=(nominal) ;;
        RunSyst|runsyst) collections=(RunSyst) ;;
        RunXSecSyst|runxsecsyst) collections=(RunXSecSyst) ;;
        MuonIDEfficiency|muonidefficiency|MuonID|muonid) collections=(MuonIDEfficiency) ;;
        TriggerEfficiency|triggerefficiency|TriggerEff|triggereff) collections=(TriggerEfficiency) ;;
        all) collections=(nominal RunSyst RunXSecSyst MuonIDEfficiency TriggerEfficiency) ;;
        *)
            echo "[ERROR] Unknown COLLECTION: $collection_selection"
            print_help
            return 2
            ;;
    esac

    local collection era status
    for collection in "${collections[@]}"; do
        for era in "${eras[@]}"; do
            run_one "$era" "$collection"
            status=$?
            if (( status != 0 )); then
                echo "[ERROR] stopped at era=$era collection=$collection"
                return "$status"
            fi
        done

        if (( make_run2_summary )); then
            merge_period "Run2" "$collection" "${run2_eras[@]}" || return $?
        fi
        if (( make_run3_summary )); then
            merge_period "Run3" "$collection" "${run3_eras[@]}" || return $?
        fi
    done

    echo
    echo "[DONE] all requested hadd jobs finished."
    return 0
}

# Unmatched wildcards should disappear rather than being passed literally to hadd.
shopt -s nullglob

hadd_main "$@"
