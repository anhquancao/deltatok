#!/bin/bash
# Keep arms queued in the main QoS; whenever the debug QoS frees a slot, move the next PENDING arm into it.
# Usage: drip.sh <bsc|jean-zay> JOBID...   (promotion order = argument order)
# LOWER_TIME=1 cuts longer walltimes to the 2 h debug cap; off by default.
host=$1; shift
case $host in
    bsc) dq=acc_debug; slots=1 ;;
    jean-zay) dq=qos_gpu_h100-dev; slots=10 ;;
    *) echo "unknown host $host"; exit 1 ;;
esac
tick="$(dirname "$0")/tick.sh"
while :; do
    out=$(ssh "$host" bash -l -s -- "$dq" "$slots" 02:00:00 "${LOWER_TIME:-0}" "$@" < "$tick" 2>/dev/null)
    if ! echo "$out" | grep -q '^WAITING'; then
        echo "$(date +%H:%M) ssh failed, retrying"
        sleep 60
        continue
    fi
    echo "$(date +%H:%M) $(echo "$out" | paste -sd '|')"
    echo "$out" | grep -q '^WAITING 0$' && break
    sleep 60
done
echo DONE
