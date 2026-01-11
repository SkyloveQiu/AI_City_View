#!/bin/bash
# 批处理监控脚本

LOG_FILE="batch_process.log"

echo "📊 批处理进度监控"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查进程是否运行
if pgrep -f "batch_run.py" > /dev/null; then
    echo "✅ 批处理进程运行中"
    echo "   PID: $(pgrep -f 'batch_run.py')"
else
    echo "⚠️  批处理进程未运行"
fi

echo ""

# 统计完成数量
if [ -f "$LOG_FILE" ]; then
    COMPLETED=$(grep -c "✅ 完成:" "$LOG_FILE" 2>/dev/null || echo 0)
    FAILED=$(grep -c "❌ 失败:" "$LOG_FILE" 2>/dev/null || echo 0)
    TOTAL=153
    
    echo "进度：$COMPLETED/$TOTAL 完成, $FAILED 失败"
    
    if [ $COMPLETED -gt 0 ]; then
        PERCENT=$((COMPLETED * 100 / TOTAL))
        echo "完成率：${PERCENT}%"
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "最近完成的图片："
    grep "✅ 完成:" "$LOG_FILE" | tail -5
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "输出文件统计："
    if [ -d "output" ]; then
        DIRS=$(find output -maxdepth 1 -type d | wc -l)
        FILES=$(find output -type f | wc -l)
        SIZE=$(du -sh output 2>/dev/null | cut -f1)
        echo "  目录数：$((DIRS - 1))"
        echo "  文件数：$FILES"
        echo "  总大小：$SIZE"
    fi
else
    echo "⚠️  日志文件不存在"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "实时日志监控："
echo "  tail -f $LOG_FILE"
echo ""
echo "停止批处理："
echo "  pkill -f batch_run.py"
