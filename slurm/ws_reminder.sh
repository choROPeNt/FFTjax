#!/usr/bin/env bash
# ws_reminder.sh -- list `ws_list` workspaces sorted by time-to-expiry,
# colour-flagging anything expiring soon.
#
# Usage:
#   ws_reminder.sh                 # full table, sorted soonest-first
#   ws_reminder.sh -a              # extra args are passed through to ws_list (e.g. -a = all filesystems)
#   ws_reminder.sh --summary       # print nothing unless something is expiring soon (for .bashrc)
#   ws_reminder.sh --input FILE    # parse a saved `ws_list` transcript instead of running it
#
# Env vars:
#   WS_WARN_DAYS  (default 14)  -- flag yellow at/below this many days remaining
#   WS_CRIT_DAYS  (default 3)   -- flag red at/below this many days remaining
#
# To get a reminder on every login, add to ~/.bashrc:
#   [[ $- == *i* ]] && ~/projects/alpha-capella/FFTjax/slurm/ws_reminder.sh --summary
set -euo pipefail

WARN_DAYS="${WS_WARN_DAYS:-14}"
CRIT_DAYS="${WS_CRIT_DAYS:-3}"
SUMMARY_ONLY=0
INPUT_FILE=""
ws_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--summary) SUMMARY_ONLY=1; shift ;;
        --input) INPUT_FILE="$2"; shift 2 ;;
        *) ws_args+=("$1"); shift ;;
    esac
done

if [[ -t 1 ]]; then
    RED=$'\033[1;31m'; YEL=$'\033[1;33m'; GRN=$'\033[0;32m'; BLD=$'\033[1m'; RST=$'\033[0m'
else
    RED=''; YEL=''; GRN=''; BLD=''; RST=''
fi

if [[ -n "$INPUT_FILE" ]]; then
    raw="$(cat "$INPUT_FILE")"
else
    raw="$(ws_list "${ws_args[@]}")"
fi

parsed="$(awk '
    /^id:/ {
        if (id != "") print id "\t" dir "\t" expd "\t" hours "\t" fs "\t" ext
        id = $2; dir = ""; expd = ""; hours = 0; fs = ""; ext = 0
        next
    }
    /remaining time/ {
        line = $0
        sub(/.*remaining time *: */, "", line)
        d = 0; h = 0
        if (match(line, /[0-9]+ days/))  { d = substr(line, RSTART, RLENGTH) + 0 }
        if (match(line, /[0-9]+ hours/)) { h = substr(line, RSTART, RLENGTH) + 0 }
        hours = d * 24 + h
        next
    }
    /expiration date/ {
        line = $0
        sub(/.*expiration date *: */, "", line)
        expd = line
        next
    }
    /workspace directory/ {
        line = $0
        sub(/.*workspace directory *: */, "", line)
        dir = line
        next
    }
    /filesystem name/ {
        line = $0
        sub(/.*filesystem name *: */, "", line)
        fs = line
        next
    }
    /available extensions/ {
        line = $0
        sub(/.*available extensions *: */, "", line)
        ext = line + 0
        next
    }
    END { if (id != "") print id "\t" dir "\t" expd "\t" hours "\t" fs "\t" ext }
' <<< "$raw" | sort -t $'\t' -k4,4n)"

if [[ -z "$parsed" ]]; then
    echo "No workspaces found (ws_list returned nothing)." >&2
    exit 0
fi

crit_list=()
warn_list=()

# scale for the extensions bar: widest bar = the largest extension count seen
ext_max=$(awk -F'\t' '{print $6}' <<< "$parsed" | sort -n | tail -1)
(( ext_max < 1 )) && ext_max=1

build_bar() {
    local n=$1 max=$2 filled empty
    (( n > max )) && n=$max
    filled=$(printf '%*s' "$n" '' | tr ' ' '#')
    empty=$(printf '%*s' $(( max - n )) '' | tr ' ' '.')
    printf '%s%s' "$filled" "$empty"
}

while IFS=$'\t' read -r id dir expd hours fs ext; do
    days=$(( hours / 24 ))
    rem_h=$(( hours % 24 ))
    if (( days <= CRIT_DAYS )); then
        crit_list+=("$id (${days}d ${rem_h}h, expires $expd)")
    elif (( days <= WARN_DAYS )); then
        warn_list+=("$id (${days}d ${rem_h}h, expires $expd)")
    fi

    if (( SUMMARY_ONLY )); then
        continue
    fi

    if (( days <= CRIT_DAYS )); then
        colour="$RED"; tag="EXPIRING"
    elif (( days <= WARN_DAYS )); then
        colour="$YEL"; tag="soon"
    else
        colour="$GRN"; tag=""
    fi

    if (( ext == 0 )); then
        ext_colour="$RED"
    elif (( ext <= 2 )); then
        ext_colour="$YEL"
    else
        ext_colour="$RST"
    fi
    bar="$(build_bar "$ext" "$ext_max")"

    printf "%s%-22s %3dd %2dh remaining   expires %-24s [%s]  %-8s%s ext %2d [%s%s%s]%s\n" \
        "$colour" "$id" "$days" "$rem_h" "$expd" "$fs" "$tag" "$RST" "$ext" "$ext_colour" "$bar" "$RST" "$RST"
done <<< "$parsed"

if (( ${#crit_list[@]} > 0 )); then
    echo
    echo "${BLD}${RED}⚠ EXPIRING WITHIN ${CRIT_DAYS} DAYS:${RST}"
    printf "  %s\n" "${crit_list[@]}"
fi

if (( ${#warn_list[@]} > 0 && (SUMMARY_ONLY || ${#crit_list[@]} == 0) )); then
    echo
    echo "${BLD}${YEL}Expiring within ${WARN_DAYS} days:${RST}"
    printf "  %s\n" "${warn_list[@]}"
fi

if (( ${#crit_list[@]} == 0 && ${#warn_list[@]} == 0 && SUMMARY_ONLY )); then
    exit 0
fi
