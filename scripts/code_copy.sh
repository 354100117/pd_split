#!/bin/bash

# 源目录（本地）
SRC_DIR="/ssd/pd/pd_infer_pipeline/whole_process"

# 目标基础路径（远程）
DST_BASE="/ssd/pd/pd_infer_pipeline"

# 节点 IP 列表
NODES=(
  "192.168.0.20"
  "192.168.0.30"
  "192.168.0.40"
  "192.168.0.50"
)

for ip in "${NODES[@]}"; do
    echo "🔄 拷贝到 nvidia@$ip ..."

    # 确保远程目录存在
    ssh -o ConnectTimeout=5 nvidia@"$ip" "mkdir -p '$DST_BASE'" || {
        echo "❌ $ip 创建目录失败"
        continue
    }

    # scp 递归拷贝（-r 必须）
    scp -r "$SRC_DIR" "nvidia@$ip:$DST_BASE/"

    if [ $? -eq 0 ]; then
        echo "✅ $ip 拷贝成功"
    else
        echo "❌ $ip 拷贝失败"
    fi

    echo "----------------------------------------"
done

