# Public Handover & Production Deployment Runbook

Comprehensive guide for deploying, maintaining, and handing over the Drought Monitoring System.

---

## 1. System Architecture & Prerequisites

### Architecture Overview
* **Frontend:** React + Vite (Port `3000` / Static build served via Nginx)
* **Backend:** FastAPI + Uvicorn (Port `8000`)
* **Reverse Proxy & SSL:** Nginx with Let's Encrypt TLS
* **Process Manager:** PM2 or Systemd

### System Requirements
* Ubuntu 20.04 / 22.04 LTS or Debian 11+
* Node.js v18+ & pnpm / npm
* Python 3.10+ & pip / venv
* Nginx 1.18+
* PM2 (`npm install -g pm2`) or Systemd

---

## 2. Git Repository & Initial Setup

```bash
# Clone repository
git clone <REPOSITORY_URL> /var/www/drought-monitor
cd /var/www/drought-monitor

# Ensure correct permissions
sudo chown -R $USER:www-data /var/www/drought-monitor
sudo chmod -R 755 /var/www/drought-monitor
```

---

## 3. Environment Configuration (`.env.production`)

Create `.env.production` in the project root / backend directory:

```env
# Application Settings
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=generate_a_secure_random_secret_key_here

# Server & Network Configuration
HOST=127.0.0.1
PORT=8000
ALLOWED_ORIGINS=https://drought-monitor.local,https://yourdomain.com

# Database / Storage (if applicable)
DATABASE_URL=sqlite:////media/DiskE/SKRIPSI/Skripsi_Nopal/data/drought.db

# API & External Services
API_V1_STR=/api/v1
MAX_WORKERS=4
```

Create frontend `.env.production`:

```env
VITE_API_BASE_URL=https://yourdomain.com/api
```

---

## 4. Backend Deployment (FastAPI)

### Option A: PM2 Process Manager (Recommended)

Create `ecosystem.config.js` in project root:

```javascript
module.exports = {
  apps: [
    {
      name: "drought-backend",
      script: "venv/bin/uvicorn",
      args: "main:app --host 127.0.0.1 --port 8000 --workers 4",
      cwd: "/var/www/drought-monitor/backend",
      interpreter: "none",
      env: {
        NODE_ENV: "production",
      },
    },
  ],
};
```

Start and save PM2 service:

```bash
# Setup Python venv and install dependencies
cd /var/www/drought-monitor/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start via PM2
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

### Option B: Systemd Service

Create `/etc/systemd/system/drought-backend.service`:

```ini
[Unit]
Description=Drought Monitor FastAPI Backend
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/drought-monitor/backend
ExecStart=/var/www/drought-monitor/backend/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --workers 4
Restart=always
RestartSec=5
EnvironmentFile=/var/www/drought-monitor/.env.production

[Install]
WantedBy=multi-user.target
```

Enable and start systemd service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable drought-backend
sudo systemctl start drought-backend
sudo systemctl status drought-backend
```

---

## 5. Frontend Deployment (React / Vite)

```bash
cd /var/www/drought-monitor/frontend

# Install dependencies and build
pnpm install
pnpm build

# Target dist directory for Nginx static serving
# Static assets built in: /var/www/drought-monitor/frontend/dist
```

---

## 6. Nginx Deployment & SSL Setup

### Install & Configure Nginx

Copy `drought-monitor.conf` to Nginx sites directory:

```bash
sudo cp /var/www/drought-monitor/drought-monitor.conf /etc/nginx/sites-available/drought-monitor.conf
sudo ln -sf /etc/nginx/sites-available/drought-monitor.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Obtain SSL Certificate (Certbot)

```bash
sudo apt update
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d yourdomain.com
```

---

## 7. Maintenance & Operations Commands

### Service Control

| Target | PM2 Command | Systemd Command |
|---|---|---|
| **Backend Status** | `pm2 status drought-backend` | `sudo systemctl status drought-backend` |
| **Backend Restart** | `pm2 restart drought-backend` | `sudo systemctl restart drought-backend` |
| **Backend Logs** | `pm2 logs drought-backend` | `sudo journalctl -u drought-backend -f` |
| **Nginx Reload** | `sudo systemctl reload nginx` | `sudo systemctl reload nginx` |

### Health Checks

```bash
# Verify API response
curl -I http://127.0.0.1:8000/health

# Verify Nginx proxy endpoint
curl -I https://yourdomain.com/api/health
```

---

## 8. Handover Checklist & Verification

- [ ] Environment variables verified in `.env.production`.
- [ ] Database migrations / seeds executed.
- [ ] PM2 / Systemd service enabled on boot.
- [ ] Nginx configuration syntax check passed (`nginx -t`).
- [ ] SSL certificate active and auto-renewal verified (`certbot renew --dry-run`).
- [ ] API documentation accessible at `/api/docs`.
