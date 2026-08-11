#!/bin/bash
# 對多個 checkpoint 各跑幾輪保留區評估,印出一張可比較的表。
#
#     bash ml/eval_sweep.sh outputs/groot_v1                 # 全部 checkpoint
#     bash ml/eval_sweep.sh outputs/groot_v1 010000 030000   # 指定幾個
#     EPISODES=30 SEEDS="0 1 2 3" bash ml/eval_sweep.sh outputs/act_v1
#
# ⚠️ **為什麼要跑好幾輪。** eval 不是決定性的:同一個 checkpoint、同一個 seed、
# 同樣的方塊位置,跑兩次會得到不同結果。實測 act_v1 的 step 5000 在保留區拿過
# 10/10 也拿過 9/10。Isaac 的物理與算繪會隨 GPU 負載改變影格到達時機,觀測就
# 跟著變,策略輸出也跟著變。單次數字拿來排名 checkpoint 會排錯。
#
# 每一輪換一個 **seed**,所以換的是方塊位置而不只是重跑同一組 —— 同時擴大取樣
# 涵蓋範圍與平均掉物理雜訊。預設 3 輪 × 20 集 = 每個 checkpoint 60 集。
#
# ⚠️ 只跑 --holdout。訓練區的成功率分辨不出「有在看影像」與「把平均軌跡背起來」,
# 保留帶(θ 在 [-35,-20] 與 [20,35],訓練資料 0 集)才有鑑別力。要訓練區的數字
# 自己單獨跑一次 eval.py 就好。
set -u

RUN=${1:?用法: bash ml/eval_sweep.sh <output_dir> [checkpoint ...]}
shift || true
EPISODES=${EPISODES:-20}
SEEDS=${SEEDS:-"0 1 2"}
CONTAINER=${CONTAINER:-omx_vla}

if [ $# -gt 0 ]; then
    CKPTS="$*"
else
    CKPTS=$(docker exec "$CONTAINER" bash -c \
        "ls /vla/$RUN/checkpoints 2>/dev/null | grep -E '^[0-9]+\$' | sort")
fi
[ -z "$CKPTS" ] && { echo "在 /vla/$RUN/checkpoints 找不到 checkpoint"; exit 1; }

echo "run      : $RUN"
echo "每輪集數 : $EPISODES     seeds: $SEEDS"
echo "checkpoint: $(echo $CKPTS | tr '\n' ' ')"
echo
printf '%-10s %-22s %8s\n' "step" "每輪(成功/總數)" "合計"
printf '%s\n' "------------------------------------------------"

for c in $CKPTS; do
    detail=""; hit=0; tot=0
    for s in $SEEDS; do
        out=$(docker exec "$CONTAINER" bash -c "
            source /opt/ros/jazzy/setup.bash
            source /workspaces/install/setup.bash
            cd /vla
            python3 ml/eval.py \
                --checkpoint $RUN/checkpoints/$c/pretrained_model \
                --episodes $EPISODES --seed $s --holdout 2>&1" \
            | grep -oP '成功 \K[0-9]+/[0-9]+' | tail -1)
        if [ -z "$out" ]; then
            detail="$detail ERR"          # 這一輪整個掛了,不要當成 0 分默默吞掉
            continue
        fi
        detail="$detail ${out}"
        hit=$((hit + ${out%/*}))
        tot=$((tot + ${out#*/}))
    done
    if [ "$tot" -gt 0 ]; then
        pct=$(awk -v h="$hit" -v t="$tot" 'BEGIN{printf "%.0f", 100*h/t}')
        printf '%-10s %-22s %5d/%-4d %3s%%\n' "$c" "$detail" "$hit" "$tot" "$pct"
    else
        printf '%-10s %-22s %8s\n' "$c" "$detail" "全失敗"
    fi
done

echo
echo "⚠️ 挑成功率最高的,不是挑最後一個 —— act_v1 實測 step 5000 比 step 15000 好。"
