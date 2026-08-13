#!/usr/bin/env bash
# =============================================================================
# HireLens — CloudWatch Alarms
# Creates alarms for API latency, errors, memory, and ML accuracy drop.
#
# Free tier: 10 alarms free. This script creates 6.
#
# Usage:
#   export AWS_DEFAULT_REGION=us-east-1
#   export EC2_INSTANCE_ID=i-0123456789abcdef0
#   export ALERT_EMAIL=your@email.com
#   bash aws/cloudwatch-alarms.sh
# =============================================================================
set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
INSTANCE_ID="${EC2_INSTANCE_ID:?Set EC2_INSTANCE_ID}"
ALERT_EMAIL="${ALERT_EMAIL:?Set ALERT_EMAIL}"

# ── Create SNS topic for email alerts ─────────────────────────────────────────
echo "Creating SNS alert topic..."
TOPIC_ARN=$(aws sns create-topic \
    --name hirelens-alerts \
    --region "${REGION}" \
    --query TopicArn \
    --output text)

aws sns subscribe \
    --topic-arn "${TOPIC_ARN}" \
    --protocol email \
    --notification-endpoint "${ALERT_EMAIL}" \
    --region "${REGION}" > /dev/null

echo "  Topic: ${TOPIC_ARN}"
echo "  Check your email (${ALERT_EMAIL}) and confirm the subscription."
echo ""

# ── Helper ────────────────────────────────────────────────────────────────────
create_alarm() {
    local name="$1"; shift
    echo "  Creating alarm: ${name}"
    aws cloudwatch put-metric-alarm --alarm-name "${name}" "$@" \
        --alarm-actions "${TOPIC_ARN}" \
        --ok-actions    "${TOPIC_ARN}" \
        --region "${REGION}"
}

echo "Creating CloudWatch alarms..."

# ── 1. High CPU (sustained) ───────────────────────────────────────────────────
create_alarm "hirelens-high-cpu" \
    --alarm-description "CPU > 85% for 10 minutes (ML inference overload)" \
    --namespace AWS/EC2 \
    --metric-name CPUUtilization \
    --dimensions Name=InstanceId,Value="${INSTANCE_ID}" \
    --statistic Average \
    --period 300 \
    --evaluation-periods 2 \
    --threshold 85 \
    --comparison-operator GreaterThanThreshold \
    --treat-missing-data missing

# ── 2. High memory usage ──────────────────────────────────────────────────────
# Requires CloudWatch Agent to be running (installed in setup-ec2.sh)
create_alarm "hirelens-high-memory" \
    --alarm-description "Memory > 90% — model may be swapping excessively" \
    --namespace HireLens/EC2 \
    --metric-name mem_used_percent \
    --dimensions Name=InstanceId,Value="${INSTANCE_ID}" \
    --statistic Average \
    --period 300 \
    --evaluation-periods 2 \
    --threshold 90 \
    --comparison-operator GreaterThanThreshold \
    --treat-missing-data missing

# ── 3. High swap usage ────────────────────────────────────────────────────────
create_alarm "hirelens-high-swap" \
    --alarm-description "Swap > 80% — ML model is hitting memory limits" \
    --namespace HireLens/EC2 \
    --metric-name swap_used_percent \
    --dimensions Name=InstanceId,Value="${INSTANCE_ID}" \
    --statistic Average \
    --period 300 \
    --evaluation-periods 1 \
    --threshold 80 \
    --comparison-operator GreaterThanThreshold \
    --treat-missing-data missing

# ── 4. Disk usage ─────────────────────────────────────────────────────────────
create_alarm "hirelens-disk-full" \
    --alarm-description "Disk > 85% — clean up Docker images or logs" \
    --namespace HireLens/EC2 \
    --metric-name disk_used_percent \
    --dimensions Name=InstanceId,Value="${INSTANCE_ID}" \
    --statistic Average \
    --period 3600 \
    --evaluation-periods 1 \
    --threshold 85 \
    --comparison-operator GreaterThanThreshold \
    --treat-missing-data missing

# ── 5. Instance status check ──────────────────────────────────────────────────
create_alarm "hirelens-instance-down" \
    --alarm-description "EC2 instance status check failed — instance may be down" \
    --namespace AWS/EC2 \
    --metric-name StatusCheckFailed \
    --dimensions Name=InstanceId,Value="${INSTANCE_ID}" \
    --statistic Maximum \
    --period 60 \
    --evaluation-periods 2 \
    --threshold 1 \
    --comparison-operator GreaterThanOrEqualToThreshold \
    --treat-missing-data breaching

# ── 6. ML accuracy drop (custom metric) ──────────────────────────────────────
# src/monitoring/monitor.py emits this metric to CloudWatch when precision_at_1 drops.
# The alarm fires if the metric appears (value=1 means retraining was triggered).
create_alarm "hirelens-model-degraded" \
    --alarm-description "Model accuracy dropped below threshold — retraining triggered" \
    --namespace HireLens/ML \
    --metric-name RetrainingTriggered \
    --dimensions Name=Model,Value=hirelens-matcher \
    --statistic Sum \
    --period 86400 \
    --evaluation-periods 1 \
    --threshold 1 \
    --comparison-operator GreaterThanOrEqualToThreshold \
    --treat-missing-data notBreaching

echo ""
echo "============================================"
echo "  CloudWatch alarms COMPLETE (6 alarms)"
echo "============================================"
echo ""
echo "View in console:"
echo "  https://${REGION}.console.aws.amazon.com/cloudwatch/home#alarmsV2:"
echo ""
echo "NOTE: Alarms 2, 3, 4 (memory/swap/disk) require the CloudWatch Agent"
echo "to be running on the EC2 instance. Run this after setup-ec2.sh."
