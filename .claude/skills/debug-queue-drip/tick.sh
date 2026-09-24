#!/bin/bash
# One pass, run on the cluster: print each arm, then move PENDING arms into the debug QoS while it has free slots.
# Args: DEBUG_QOS SLOTS MAX_WALL(HH:MM:SS) LOWER_TIME(0|1) JOBID...
# Out: "<id> <state> <qos> <timelimit>" per arm, MOVED/REFUSED/SKIP per action, then "WAITING <n>".
dq=$1 slots=$2 maxwall=$3 lower=$4; shift 4

secs() {  # slurm time (M:S, H:M:S, D-H:M:S) -> seconds
    echo "$1" | awk -F'[-:]' '{ if (index($0, "-")) { d=$1; h=$2; m=$3; s=$4 }
        else if (NF == 3) { d=0; h=$1; m=$2; s=$3 } else { d=0; h=0; m=$1; s=$2 }
        print ((d*24 + h)*60 + m)*60 + s }'
}

used=$(squeue -u "$USER" -q "$dq" -h -o %i | wc -l)   # pending + running both hold a slot
waiting=0
for j in "$@"; do
    read -r st q tl <<<"$(squeue -j "$j" -h -o '%T %q %l' 2>/dev/null)"
    echo "$j ${st:-GONE} ${q:--} ${tl:--}"
    if [ "$st" != PENDING ] || [ "$q" = "$dq" ]; then
        continue
    fi
    if [ "$(secs "$tl")" -gt "$(secs "$maxwall")" ]; then
        if [ "$lower" != 1 ]; then
            echo "SKIP $j timelimit $tl > $maxwall"
            continue
        fi
        scontrol update JobId="$j" TimeLimit="$maxwall"   # users may only lower it
    fi
    if [ "$used" -ge "$slots" ]; then
        waiting=$((waiting + 1))
        continue
    fi
    if msg=$(scontrol update JobId="$j" QOS="$dq" 2>&1); then
        echo "MOVED $j"
        used=$((used + 1))
    else
        echo "REFUSED $j $msg"
        waiting=$((waiting + 1))
    fi
done
echo "WAITING $waiting"
