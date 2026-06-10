#!/usr/bin/env bash
# =============================================================================
# HireLens — S3 Setup Script
# Creates two S3 buckets and configures them for HireLens.
#
# Free tier: 5 GB storage, 20,000 GET, 2,000 PUT requests/month.
# For a portfolio project this stays free indefinitely.
#
# Usage:
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   export AWS_DEFAULT_REGION=us-east-1
#   bash aws/s3-setup.sh
# =============================================================================
set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
# Bucket names must be globally unique — suffix with your AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
RESUMES_BUCKET="hirelens-resumes-${ACCOUNT_ID}"
MODELS_BUCKET="hirelens-models-${ACCOUNT_ID}"

echo "Account:         ${ACCOUNT_ID}"
echo "Region:          ${REGION}"
echo "Resumes bucket:  ${RESUMES_BUCKET}"
echo "Models bucket:   ${MODELS_BUCKET}"
echo ""

# ── Helper: create bucket ─────────────────────────────────────────────────────
create_bucket() {
    local name="$1"
    if aws s3api head-bucket --bucket "${name}" 2>/dev/null; then
        echo "  Bucket '${name}' already exists — skipping creation."
    else
        if [ "${REGION}" = "us-east-1" ]; then
            aws s3api create-bucket --bucket "${name}" --region "${REGION}"
        else
            aws s3api create-bucket --bucket "${name}" --region "${REGION}" \
                --create-bucket-configuration LocationConstraint="${REGION}"
        fi
        echo "  Created: ${name}"
    fi
}

# ── 1. Create buckets ─────────────────────────────────────────────────────────
echo "[1/5] Creating S3 buckets..."
create_bucket "${RESUMES_BUCKET}"
create_bucket "${MODELS_BUCKET}"

# ── 2. Block all public access ────────────────────────────────────────────────
echo "[2/5] Blocking public access..."
for bucket in "${RESUMES_BUCKET}" "${MODELS_BUCKET}"; do
    aws s3api put-public-access-block \
        --bucket "${bucket}" \
        --public-access-block-configuration \
          "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
done

# ── 3. Enable server-side encryption ─────────────────────────────────────────
echo "[3/5] Enabling AES-256 encryption..."
for bucket in "${RESUMES_BUCKET}" "${MODELS_BUCKET}"; do
    aws s3api put-bucket-encryption \
        --bucket "${bucket}" \
        --server-side-encryption-configuration '{
          "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
              "SSEAlgorithm": "AES256"
            },
            "BucketKeyEnabled": true
          }]
        }'
done

# ── 4. CORS for resume uploads bucket ────────────────────────────────────────
# Required so the React frontend can upload PDFs directly to S3 (if desired).
echo "[4/5] Setting CORS on resumes bucket..."
aws s3api put-bucket-cors \
    --bucket "${RESUMES_BUCKET}" \
    --cors-configuration file://aws/s3-cors.json

# ── 5. Lifecycle rule — auto-delete uploaded resumes after 30 days ───────────
# Keeps storage within free tier even if resumes accumulate.
echo "[5/5] Setting lifecycle policy (delete after 30 days)..."
aws s3api put-bucket-lifecycle-configuration \
    --bucket "${RESUMES_BUCKET}" \
    --lifecycle-configuration '{
      "Rules": [{
        "ID": "auto-delete-uploads",
        "Status": "Enabled",
        "Filter": {"Prefix": "uploads/"},
        "Expiration": {"Days": 30}
      }]
    }'

echo ""
echo "============================================"
echo "  S3 setup COMPLETE"
echo "============================================"
echo ""
echo "Add these to your .env.prod:"
echo "  AWS_S3_RESUMES_BUCKET=${RESUMES_BUCKET}"
echo "  AWS_S3_MODELS_BUCKET=${MODELS_BUCKET}"
echo "  AWS_DEFAULT_REGION=${REGION}"
echo ""
echo "Upload your trained model to S3:"
echo "  aws s3 sync models/fine_tuned/hirelens_matcher \\"
echo "    s3://${MODELS_BUCKET}/models/hirelens_matcher/"
