#!/bin/bash
# run_api_robust.sh
# 更健壮的版本，包含错误处理和日志

set -e  # 遇到错误时退出

# 配置
API_URL="http://192.168.0.10:8000/api/run"
LOG_FILE="/tmp/api_request_$(date +%Y%m%d_%H%M%S).log"
REQUEST_DATA='{
    "prompts": [
        "Short prompt 1",
        "Short prompt 1",
        "Short prompt 1",
        "Short prompt 1"
    ],
    "max_new_tokens": 512
}'

# 记录日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 开始执行
log "开始发送API请求到: $API_URL"
log "请求数据: $REQUEST_DATA"

# 发送请求并记录
response=$(curl -s -X POST "$API_URL" \
     -H "Content-Type: application/json" \
     -d "$REQUEST_DATA" \
     -w "\nHTTP状态码: %{http_code}\n" \
     --connect-timeout 30 \
     --max-time 60)

# 检查返回状态
http_code=$(echo "$response" | grep "HTTP状态码" | awk '{print $NF}')
response_body=$(echo "$response" | sed '/HTTP状态码/d')

log "HTTP状态码: $http_code"

if [ "$http_code" -eq 200 ]; then
    log "请求成功!"
    echo "响应内容:"
    echo "$response_body" | jq '.' 2>/dev/null || echo "$response_body"
else
    log "请求失败! HTTP状态码: $http_code"
    echo "错误响应:"
    echo "$response_body"
    exit 1
fi

log "请求完成"
