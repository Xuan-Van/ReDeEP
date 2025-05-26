#!/bin/bash

# 定义每个进程使用的 CUDA 设备
declare -a CUDA_DEVICES=("0,1" "2,3" "4,5" "6,7")

# 定义要执行的起始和结束编号
START=0
END=32

# 定义每批执行的进程数
BATCH_SIZE=4

# 定义汇总日志文件
LOG_FILE="log.txt"

# 清空之前的汇总日志文件
> "$LOG_FILE"

# 循环执行任务
for ((i=START; i<=END; i+=BATCH_SIZE)); do
    echo "Starting batch: $i to $((i+BATCH_SIZE-1))"
    for ((j=0; j<BATCH_SIZE; j++)); do
        if [ $((i+j)) -le $END ]; then
            # 定义每个进程的日志文件
            LOG="test/chunk/log_$((i+j)).txt"
            # 启动进程并计算执行时间
            (time CUDA_VISIBLE_DEVICES=${CUDA_DEVICES[j]} python src/detect.py --layer $((i+j)) > "$LOG" 2>&1) 2>> "$LOG_FILE" &
        fi
    done
    # 等待当前批次的所有进程结束
    wait
    echo "Batch $i to $((i+BATCH_SIZE-1)) completed."
done

echo "All tasks completed."