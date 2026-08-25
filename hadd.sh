#!/usr/bin/env bash

# NIsoMuon Run-2/Run-3 ROOT-file merger
# =====================================
#
# This script is safe to run either as
#
#   source hadd.sh Run3 all
#
# or
#
#   bash hadd.sh Run3 all
#
# IMPORTANT:
#   - There is no `exit` in this script, so sourcing it will not close your shell/SSH session.
#   - There is no `set -e`, `pipefail`, process substitution, or shell pipeline.
#   - Missing input files are simply skipped.
#
# Fixed base directory:
#   /data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon
#
# Usage:
#   source hadd.sh ERA_SELECTION [COLLECTION]
#
# ERA_SELECTION:
#   2016preVFP, 2016postVFP, 2017, 2018,
#   2022, 2022EE, 2023, 2023BPix,
#   Run2, Run3, Run2+3, all
#
# COLLECTION:
#   nominal      : <BASE>/<era>
#   RunSyst      : <BASE>/RunSyst/<era>
#   RunXSecSyst  : <BASE>/RunXSecSyst/<era>
#   all          : all three
#
# Process grouping:
#   data:
#     Run2        -> SingleMuon_*
#     2022/2022EE -> Muon_*
#     2023/2023BPix -> Muon0_* + Muon1_*
#     Per-era merged output is <BASE>/<era>/data.root (no DATA/ subdirectory).
#
# Period merge:
#   source hadd.sh Run2
#     -> <BASE>/Run2/*.root
#   source hadd.sh Run3
#     -> <BASE>/Run3/*.root
#   For RunSyst/RunXSecSyst the same structure is used under the collection:
#     <BASE>/RunSyst/Run2/, <BASE>/RunSyst/Run3/, etc.
#
#   QCD:
#     all QCD_Pt-*_MuEnriched
#
#   DY:
#     aMC -> DYJets + DYJets10to50
#     MG  -> DYJets_MG + DYJets10to50_MG
#     These are NEVER mixed.
#
#   ST:
#     Run2 -> SingleTop_*
#     Run3 -> ST_*
#
#   ttbar:
#     TTLL_powheg + TTLJ_powheg + TTJJ_powheg ONLY.
#     TTLL tune/hdamp/mtop/ext samples are excluded.
#
#   Others:
#     Run2 -> WJets_MG, TTG, TTWToLNu, TTZToLLNuNu
#     Run3 -> TTG_PTG10to100, TTG_PTG100to200, TTG_PTG200toInf,
#             TTZ_NoFullyHad, WJets_MG
#
# Examples:
#   source hadd.sh 2023
#   source hadd.sh Run3 nominal
#   source hadd.sh Run2 RunSyst
#   source hadd.sh all RunXSecSyst
#   source hadd.sh Run3 all
#   source hadd.sh all all

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
    echo "  nominal, RunSyst, RunXSecSyst, all"
    echo
    echo "Examples:"
    echo "  source hadd.sh Run3 all"
    echo "  source hadd.sh all all"
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
        rm -f "$output"
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
        2016preVFP|2016postVFP|2017|2018)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
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

merge_ttbar() {
    local dir="$1"
    local inputs=(
        "$dir/Skim_NIsoMuon_TTLL_powheg.root"
        "$dir/Skim_NIsoMuon_TTLJ_powheg.root"
        "$dir/Skim_NIsoMuon_TTJJ_powheg.root"
    )

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

merge_others() {
    local era="$1"
    local dir="$2"
    local inputs=()

    if is_run2 "$era"; then
        inputs=(
            "$dir/Skim_NIsoMuon_WJets_MG.root"
            "$dir/Skim_NIsoMuon_TTG.root"
            "$dir/Skim_NIsoMuon_TTWToLNu.root"
            "$dir/Skim_NIsoMuon_TTZToLLNuNu.root"
            "$dir/Skim_NIsoMuon_ttZToLLNuNu.root"
        )
    else
        inputs=(
            "$dir/Skim_NIsoMuon_TTG_PTG10to100.root"
            "$dir/Skim_NIsoMuon_TTG_PTG100to200.root"
            "$dir/Skim_NIsoMuon_TTG_PTG200toInf.root"
            "$dir/Skim_NIsoMuon_TTZ_NoFullyHad.root"
            "$dir/Skim_NIsoMuon_WJets_MG.root"
        )
    fi

    run_hadd "$dir/NIsoMuon_Others.root" "${inputs[@]}"
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

    run_hadd "$dir/NIsoMuon_AllMC.root" "${allmc_inputs[@]}"
    run_hadd "$dir/NIsoMuon_QCDTop.root" "${qcdtop_inputs[@]}"
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

    # Data exists only in the nominal collection.
    if [[ "$collection" == "nominal" ]]; then
        merge_data "$era" "$dir"
    fi

    merge_qcd "$dir"
    merge_st "$era" "$dir"
    merge_ttbar "$dir"
    merge_top "$dir"
    merge_dy "$dir"
    merge_others "$era" "$dir"

    # RunXSecSyst is process-specific and does not need these mixed summary files.
    if [[ "$collection" != "RunXSecSyst" ]]; then
        merge_summary "$dir"
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
        if [[ -f "$ERA_DIR/$filename" ]]; then
            inputs+=( "$ERA_DIR/$filename" )
        fi
    done

    run_hadd "$output" "${inputs[@]}"
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

    # Data is only produced for the nominal collection.
    if [[ "$collection" == "nominal" ]]; then
        merge_period_file "$period" "$collection" "data.root" "${eras[@]}"
    fi

    merge_period_file "$period" "$collection" "NIsoMuon_QCD_Inclusive.root"       "${eras[@]}"
    merge_period_file "$period" "$collection" "NIsoMuon_ST.root"                  "${eras[@]}"
    merge_period_file "$period" "$collection" "NIsoMuon_tt.root"                  "${eras[@]}"
    merge_period_file "$period" "$collection" "NIsoMuon_Top.root"                 "${eras[@]}"
    merge_period_file "$period" "$collection" "NIsoMuon_DYJets_Inclusive.root"    "${eras[@]}"
    merge_period_file "$period" "$collection" "NIsoMuon_DYJets_MG_Inclusive.root" "${eras[@]}"
    merge_period_file "$period" "$collection" "NIsoMuon_Others.root"              "${eras[@]}"

    # RunXSecSyst intentionally has no mixed summary files.
    if [[ "$collection" != "RunXSecSyst" ]]; then
        merge_period_file "$period" "$collection" "NIsoMuon_AllMC.root"            "${eras[@]}"
        merge_period_file "$period" "$collection" "NIsoMuon_QCDTop.root"           "${eras[@]}"
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
        nominal|Nominal|"")
            collections=(nominal)
            ;;
        RunSyst|runsyst)
            collections=(RunSyst)
            ;;
        RunXSecSyst|runxsecsyst)
            collections=(RunXSecSyst)
            ;;
        all)
            collections=(nominal RunSyst RunXSecSyst)
            ;;
        *)
            echo "[ERROR] Unknown COLLECTION: $collection_selection"
            print_help
            return 2
            ;;
    esac

    local collection
    local era
    local status=0

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
            merge_period "Run2" "$collection" "${run2_eras[@]}"
            status=$?
            if (( status != 0 )); then
                echo "[ERROR] Run2 period merge failed for collection=$collection"
                return "$status"
            fi
        fi

        if (( make_run3_summary )); then
            merge_period "Run3" "$collection" "${run3_eras[@]}"
            status=$?
            if (( status != 0 )); then
                echo "[ERROR] Run3 period merge failed for collection=$collection"
                return "$status"
            fi
        fi
    done

    echo
    echo "[DONE] all requested hadd jobs finished."
    return 0
}

# nullglob makes an unmatched wildcard expand to an empty array, instead of
# passing the literal '*' string to hadd.
shopt -s nullglob

# Calling a function with `return` is safe both when the file is sourced and
# when it is executed with bash.  There is intentionally no `exit` here.
hadd_main "$@"

