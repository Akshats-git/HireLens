#!/usr/bin/env bash
# =============================================================================
# HireLens — EC2 Bootstrap Script
# Run ONCE on a fresh Ubuntu 22.04 LTS t2.micro instance.
#
# Usage:
#   scp -i your-key.pem aws/setup-ec2.sh ubuntu@<EC2_IP>:~
#   ssh -i your-key.pem ubuntu@<EC2_IP> "bash setup-ec2.sh [your-domain.com]"
#
# What this does:
#   1. Creates a 4 GB swap file (t2.micro has 1 GB RAM — needed for ML model)
#   2. Installs Docker, nginx, certbot, CloudWatch agent, awscli
#   3. Configures nginx as reverse proxy (port 80 → Docker containers)
#   4. Optionally issues a Let's Encrypt SSL cert (pass your domain as $1)
#   5. Creates a systemd service so Docker containers restart on reboot
# =============================================================================
set -euo pipefail

DOMAIN="${1:-}"
APP_DIR="/opt/hirelens"

echo "============================================"
echo "  HireLens EC2 Bootstrap"
echo "  Domain: ${DOMAIN:-none (HTTP only)}"
echo "============================================"

# ── 1. System update ──────────────────────────────────────────────────────────
echo "[1/8] Updating system packages..."
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -qq

# ── 2. Install dependencies ───────────────────────────────────────────────────
echo "[2/8] Installing dependencies..."
sudo apt-get install -y -qq \
    git curl unzip jq \
    nginx certbot python3-certbot-nginx \
    awscli \
    htop

# ── 3. Swap file ─────────────────────────────────────────────────────────────
# t2.micro = 1 GB RAM. ML model (PyTorch + sentence-transformers) needs ~1.5 GB.
# The 4 GB swap lets the model load — inference will be ~3-5s (vs 0.5s on 2 GB).
# For production with real traffic, upgrade to t3.small (2 GB, ~$15/mo).
echo "[3/8] Creating 4 GB swap file..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab > /dev/null
    # Prefer RAM; only spill to swap as last resort
    echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf > /dev/null
    sudo sysctl -p > /dev/null
    echo "  Swap: $(free -h | awk '/Swap/ {print $2}')"
else
    echo "  Swap already exists, skipping."
fi

# ── 4. Docker ─────────────────────────────────────────────────────────────────
echo "[4/8] Installing Docker..."
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker ubuntu
    sudo systemctl enable docker
    sudo systemctl start docker
else
    echo "  Docker already installed: $(docker --version)"
fi

# Docker Compose plugin
if ! docker compose version &>/dev/null 2>&1; then
    sudo apt-get install -y -qq docker-compose-plugin
fi

# ── 5. CloudWatch Agent ───────────────────────────────────────────────────────
echo "[5/8] Installing CloudWatch Agent..."
if ! command -v amazon-cloudwatch-agent-ctl &>/dev/null; then
    wget -q https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
    sudo dpkg -i amazon-cloudwatch-agent.deb
    rm amazon-cloudwatch-agent.deb
else
    echo "  CloudWatch Agent already installed."
fi

# ── 6. App directory ──────────────────────────────────────────────────────────
echo "[6/8] Creating app directory at ${APP_DIR}..."
sudo mkdir -p "${APP_DIR}"
sudo chown ubuntu:ubuntu "${APP_DIR}"

# ── 7. nginx config ───────────────────────────────────────────────────────────
echo "[7/8] Configuring nginx..."
sudo tee /etc/nginx/sites-available/hirelens > /dev/null <<'NGINX_CONF'
# ── Rate limiting zone (shared memory) ───────────────────────────────────────
limit_req_zone $binary_remote_addr zone=api:10m rate=20r/s;

server {
    listen 80;
    server_name _;

    # Frontend (React — Docker container on port 3000)
    location / {
        proxy_pass         http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 30s;
    }

    # Backend API (FastAPI — Docker container on port 8000)
    location /api/ {
        limit_req zone=api burst=40 nodelay;
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        client_max_body_size 11M;
        proxy_read_timeout   120s;   # bulk-analyze can take up to 30s
        proxy_send_timeout   120s;
    }

    # Health check — no rate limit
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_set_header Host $host;
        access_log off;
    }

    # Block common exploit paths
    location ~* \.(env|git|htaccess|htpasswd)$ { deny all; }
}
NGINX_CONF

sudo ln -sf /etc/nginx/sites-available/hirelens /etc/nginx/sites-enabled/hirelens
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl reload nginx

# ── SSL (optional) ────────────────────────────────────────────────────────────
if [ -n "$DOMAIN" ]; then
    echo "  Requesting SSL certificate for ${DOMAIN}..."
    sudo certbot --nginx \
        -d "${DOMAIN}" \
        --non-interactive \
        --agree-tos \
        --email "admin@${DOMAIN}" \
        --redirect \
        --no-eff-email
    # Auto-renew twice daily (certbot recommends this)
    (crontab -l 2>/dev/null; \
     echo "0 3,15 * * * certbot renew --quiet && systemctl reload nginx") \
        | sort -u | crontab -
    echo "  SSL configured. Certificate auto-renews every 90 days."
fi

# ── 8. Systemd service ────────────────────────────────────────────────────────
echo "[8/8] Creating systemd service..."
sudo tee /etc/systemd/system/hirelens.service > /dev/null <<EOF
[Unit]
Description=HireLens Application Stack
After=network-online.target docker.service
Requires=docker.service
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/.env.prod
ExecStartPre=/usr/bin/docker compose -f ${APP_DIR}/docker-compose.prod.yml pull
ExecStart=/usr/bin/docker compose -f ${APP_DIR}/docker-compose.prod.yml up -d
ExecStop=/usr/bin/docker compose -f ${APP_DIR}/docker-compose.prod.yml down
StandardOutput=journal
StandardError=journal
SyslogIdentifier=hirelens
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable hirelens

echo ""
echo "============================================"
echo "  Bootstrap COMPLETE"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Copy files to EC2:"
echo "     scp -i key.pem docker-compose.prod.yml ubuntu@${DOMAIN:-<EC2_IP>}:${APP_DIR}/"
echo "     scp -i key.pem aws/.env.prod         ubuntu@${DOMAIN:-<EC2_IP>}:${APP_DIR}/.env.prod"
echo "     scp -i key.pem aws/cloudwatch-agent.json ubuntu@${DOMAIN:-<EC2_IP>}:~/"
echo ""
echo "  2. Configure CloudWatch agent:"
echo "     sudo amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -s -c file:cloudwatch-agent.json"
echo ""
echo "  3. Start the app:"
echo "     sudo systemctl start hirelens"
echo "     sudo systemctl status hirelens"
echo ""
echo "  4. Verify:"
echo "     curl http://${DOMAIN:-localhost}/health"
