#!/bin/bash
# long_seq_test.sh
# 长序列输出测试脚本

# 配置
API_URL="http://192.168.0.10:8000/api/run"
MAX_NEW_TOKENS=32
OUTPUT_DIR="./test_outputs"
LOG_FILE="${OUTPUT_DIR}/test_$(date +%Y%m%d_%H%M%S).log"
BATCH_SIZE=4

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 日志函数
log() {
    local message="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" | tee -a "$LOG_FILE"
}

# 长提示词（保持完全一致，便于 decode batch）
PROMPT=$(cat <<'EOF'
You are designing a real-time inference pipeline for the OPT-2.7B model on an NVIDIA Jetson AGX Orin with 32GB unified memory. The system must handle up to 20 concurrent users, each submitting prompts of variable length (50–512 tokens) and expecting streaming output. Due to hardware constraints, you cannot use vLLM or TGI; instead, you must build a minimal custom server using Hugging Face Transformers and PyTorch. Describe in detail: (1) how you would implement dynamic batching to improve GPU utilization during the decoding phase, (2) strategies to avoid out-of-memory errors—especially considering shared CPU/GPU memory architecture, (3) whether you'd separate prefill and decode stages, and why, (4) how you'd ensure that requests submitted earlier finish before later ones (FIFO guarantee), and (5) what metrics you'd log to monitor latency, throughput, and memory pressure. Assume quantization (e.g., 8-bit) is already applied. Focus on practical, implementable solutions rather than theoretical ideals.
EOF
)

# 测试函数
run_test() {
    local test_name="Batch_${BATCH_SIZE}"
    
    log "开始测试: $test_name"
    log "提示词长度: ${#PROMPT} 字符"
    log "批量大小: $BATCH_SIZE"
    log "最大生成长度: $MAX_NEW_TOKENS tokens"
    
    # 构建请求数据
    REQUEST_DATA=$(jq -n \
        --arg prompt "$PROMPT" \
        --argjson max_tokens "$MAX_NEW_TOKENS" \
        --argjson batch "$BATCH_SIZE" \
        '{
            "prompts": (reduce range(0; $batch) as $i ([]; . + [$prompt])),
            "max_new_tokens": $max_tokens
        }')
    
    # 发送请求
    START_TIME=$(date +%s%3N)
    
    response=$(curl -s -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -d "$REQUEST_DATA" \
        -w "\nHTTP状态码: %{http_code}\n时间统计: 总耗时=%{time_total}s DNS解析=%{time_namelookup}s 连接=%{time_connect}s TLS=%{time_appconnect}s 准备=%{time_pretransfer}s 首字节=%{time_starttransfer}s 传输=%{time_total}s" \
        --connect-timeout 60 \
        --max-time 300)
    
    END_TIME=$(date +%s%3N)
    ELAPSED_MS=$((END_TIME - START_TIME))
    
    # 解析响应
    http_code=$(echo "$response" | grep "HTTP状态码" | awk '{print $NF}')
    curl_timing=$(echo "$response" | grep "时间统计")
    response_body=$(echo "$response" | sed '/HTTP状态码/d; /时间统计/d')
    
    # 保存响应
    echo "$response_body" > "${OUTPUT_DIR}/${test_name}_response.json"
    
    log "测试 $test_name 完成"
    log "HTTP状态码: $http_code"
    log "总耗时: ${ELAPSED_MS}ms"
    log "CURL计时: $curl_timing"
    
    # 检查响应
    if [ "$http_code" -eq 200 ]; then
        log "✅ 测试 $test_name 成功"
        # 提取生成文本（如果响应是JSON）
        if echo "$response_body" | jq -e . >/dev/null 2>&1; then
            generated_text=$(echo "$response_body" | jq -r '.generated_text // .text // .response // .output // .')
            log "生成文本长度: ${#generated_text} 字符"
            echo "$generated_text" > "${OUTPUT_DIR}/${test_name}_generated.txt"
        else
            log "响应不是有效的JSON，直接保存"
        fi
    else
        log "❌ 测试 $test_name 失败"
        echo "错误响应: $response_body" >> "${OUTPUT_DIR}/${test_name}_error.log"
    fi
    
    log "----------------------------------------"
}

# 主函数
main() {
    log "开始长序列输出测试"
    log "API端点: $API_URL"
    log "输出目录: $OUTPUT_DIR"
    log "总测试数: 1"
    run_test
    
    log "所有测试完成"
    log "结果保存在: $OUTPUT_DIR"
}

# 运行主函数
main
