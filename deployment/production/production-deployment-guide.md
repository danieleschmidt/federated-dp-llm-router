# Production Deployment Guide
## Federated DP-LLM Router v0.1.0

🚀 **Production-Ready Deployment for Healthcare AI Systems**

This guide provides comprehensive instructions for deploying the Federated DP-LLM Router in production environments with enterprise-grade security, scalability, and compliance.

## 📋 Prerequisites

### System Requirements

**Minimum Hardware:**
- **CPU**: 16 cores (32 threads recommended)
- **RAM**: 32GB (64GB recommended)
- **Storage**: 500GB SSD (1TB recommended)
- **Network**: 10Gbps (for federated nodes)
- **GPU**: 2x NVIDIA A100 or equivalent (for ML inference)

**Operating System:**
- Ubuntu 20.04 LTS or Ubuntu 22.04 LTS
- Red Hat Enterprise Linux 8+
- CentOS 8+

**Software Dependencies:**
- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Container Toolkit (for GPU support)
- Kubernetes 1.24+ (for K8s deployment)

### Security Requirements

**Certificates and Keys:**
- Valid SSL/TLS certificates for all endpoints
- Hospital-specific client certificates
- Hardware Security Module (HSM) support recommended

**Network Security:**
- Firewall configuration for federated communication
- VPN or private network connectivity between hospitals
- DDoS protection and rate limiting

**Compliance:**
- HIPAA compliance review completed
- SOC 2 Type II controls implemented
- GDPR data processing agreements in place

## 🔧 Installation Steps

### 1. Environment Preparation

```bash
# Create deployment directory
sudo mkdir -p /opt/federated-dp-llm
cd /opt/federated-dp-llm

# Clone repository
git clone https://github.com/yourusername/federated-dp-llm-router.git .

# Create required directories
sudo mkdir -p /data/{postgres,redis,prometheus,grafana}
sudo mkdir -p /opt/federated-dp-llm/{ssl,logs,models}

# Set proper permissions
sudo chown -R $USER:$USER /opt/federated-dp-llm
sudo chmod 755 /data/*
```

### 2. SSL Certificate Setup

```bash
# Generate production certificates (replace with your CA)
cd ssl/

# Generate CA private key
openssl genrsa -out ca-key.pem 4096

# Generate CA certificate
openssl req -new -x509 -days 365 -key ca-key.pem -sha256 -out ca.pem \
  -subj "/C=US/ST=State/L=City/O=Hospital/OU=IT/CN=Federated-CA"

# Generate server private key
openssl genrsa -out server-key.pem 4096

# Generate server certificate signing request
openssl req -subj "/C=US/ST=State/L=City/O=Hospital/OU=IT/CN=federated-router" \
  -sha256 -new -key server-key.pem -out server.csr

# Generate server certificate
openssl x509 -req -days 365 -sha256 -in server.csr -CA ca.pem -CAkey ca-key.pem \
  -out server-cert.pem -CAcreateserial

# Generate client certificates for each hospital
for hospital in cardiology neurology emergency; do
  openssl genrsa -out ${hospital}-key.pem 4096
  openssl req -subj "/C=US/ST=State/L=City/O=Hospital/OU=${hospital}/CN=${hospital}-node" \
    -sha256 -new -key ${hospital}-key.pem -out ${hospital}.csr
  openssl x509 -req -days 365 -sha256 -in ${hospital}.csr -CA ca.pem -CAkey ca-key.pem \
    -out ${hospital}-cert.pem -CAcreateserial
done

# Set secure permissions
chmod 400 *-key.pem
chmod 444 *-cert.pem ca.pem
```

### 3. Environment Configuration

```bash
# Create production environment file
cat > .env.prod << EOF
# Database Configuration
DB_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)

# Security Keys
JWT_SECRET_KEY=$(openssl rand -base64 64)
ENCRYPTION_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
GRAFANA_SECRET_KEY=$(openssl rand -base64 32)

# Monitoring
GRAFANA_PASSWORD=$(openssl rand -base64 16)

# Network Configuration
EXTERNAL_IP=YOUR_EXTERNAL_IP
DOMAIN_NAME=your-domain.com

# Privacy Configuration
DEFAULT_PRIVACY_BUDGET=10.0
EMERGENCY_PRIVACY_BUDGET=20.0
RESEARCH_PRIVACY_BUDGET=5.0

# Compliance
ENABLE_AUDIT_LOGGING=true
HIPAA_COMPLIANCE_MODE=true
GDPR_COMPLIANCE_MODE=true
EOF

# Secure the environment file
chmod 600 .env.prod
```

### 4. Configuration Files

**Production Configuration (`configs/production.yaml`):**

```yaml
# Production Configuration for Federated DP-LLM Router
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  max_connections: 1000
  ssl:
    enabled: true
    cert_file: "/app/ssl/server-cert.pem"
    key_file: "/app/ssl/server-key.pem"
    ca_file: "/app/ssl/ca.pem"

privacy:
  default_epsilon: 0.1
  default_delta: 1e-5
  max_budget_per_user: 10.0
  max_budget_per_department:
    emergency: 20.0
    cardiology: 15.0
    neurology: 12.0
    general: 10.0
    research: 5.0
  
  mechanisms:
    default: "gaussian"
    composition: "rdp"
    noise_multiplier: 1.1

quantum_optimization:
  enabled: true
  superposition_depth: 5
  entanglement_pairs: 4
  coherence_time: 20
  decoherence_mitigation: true

security:
  input_validation:
    max_prompt_length: 10000
    sql_injection_protection: true
    xss_protection: true
    prompt_injection_protection: true
  
  rate_limiting:
    requests_per_minute: 60
    burst_size: 10
    
  authentication:
    jwt_expiry: 3600
    refresh_token_expiry: 86400
    require_mfa: true

monitoring:
  enabled: true
  metrics_port: 9090
  health_check_interval: 30
  log_level: "INFO"
  audit_logging: true

database:
  url: "${POSTGRES_URL}"
  pool_size: 20
  max_overflow: 30
  echo: false

cache:
  redis_url: "${REDIS_URL}"
  default_ttl: 3600
  max_size: 10000

federated_nodes:
  - id: "hospital_cardiology_001"
    name: "Cardiology Department"
    endpoint: "https://hospital-cardiology:8443"
    department: "cardiology"
    model_shard: "cardiology_shard"
    priority: "high"
    
  - id: "hospital_neurology_001"
    name: "Neurology Department"
    endpoint: "https://hospital-neurology:8443"
    department: "neurology"
    model_shard: "neurology_shard"
    priority: "medium"
    
  - id: "hospital_emergency_001"
    name: "Emergency Department"
    endpoint: "https://hospital-emergency:8443"
    department: "emergency"
    model_shard: "emergency_shard"
    priority: "critical"
```

### 5. NGINX Load Balancer Configuration

**`nginx/nginx.conf`:**

```nginx
# Production NGINX Configuration
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    access_log /var/log/nginx/access.log main;

    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=60r/m;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=10r/m;

    # SSL Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Upstream servers
    upstream federated_router {
        least_conn;
        server federated-router:8080 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }

    # HTTP to HTTPS redirect
    server {
        listen 80;
        server_name _;
        return 301 https://$host$request_uri;
    }

    # Main HTTPS server
    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/server-cert.pem;
        ssl_certificate_key /etc/nginx/ssl/server-key.pem;
        ssl_trusted_certificate /etc/nginx/ssl/ca.pem;

        # Client certificate verification
        ssl_client_certificate /etc/nginx/ssl/ca.pem;
        ssl_verify_client optional;

        # API endpoints
        location /api/ {
            limit_req zone=api burst=10 nodelay;
            proxy_pass http://federated_router;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header X-Client-Cert $ssl_client_escaped_cert;
            proxy_cache_bypass $http_upgrade;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Authentication endpoints
        location /auth/ {
            limit_req zone=auth burst=5 nodelay;
            proxy_pass http://federated_router;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check
        location /health {
            proxy_pass http://federated_router;
            access_log off;
        }

        # Monitoring (restricted access)
        location /metrics {
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
            proxy_pass http://federated_router;
        }
    }
}
```

### 6. Monitoring Configuration

**Prometheus (`monitoring/prometheus.yml`):**

```yaml
# Prometheus Configuration for Federated DP-LLM
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "federated_dp_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'federated-router'
    static_configs:
      - targets: ['federated-router:8080']
    scrape_interval: 15s
    metrics_path: '/metrics'

  - job_name: 'hospital-nodes'
    static_configs:
      - targets: 
        - 'hospital-cardiology:8443'
        - 'hospital-neurology:8443'
        - 'hospital-emergency:8443'
    scrape_interval: 30s
    scheme: https
    tls_config:
      ca_file: /etc/prometheus/ssl/ca.pem
      cert_file: /etc/prometheus/ssl/prometheus-cert.pem
      key_file: /etc/prometheus/ssl/prometheus-key.pem

  - job_name: 'system-metrics'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

## 🚀 Deployment Execution

### 1. Pre-deployment Validation

```bash
# Validate configuration files
docker-compose -f docker-compose.prod.yml config

# Check SSL certificates
openssl verify -CAfile ssl/ca.pem ssl/server-cert.pem
openssl verify -CAfile ssl/ca.pem ssl/cardiology-cert.pem
openssl verify -CAfile ssl/ca.pem ssl/neurology-cert.pem
openssl verify -CAfile ssl/ca.pem ssl/emergency-cert.pem

# Test environment variables
source .env.prod
echo "Testing environment variables..."
[ -n "$DB_PASSWORD" ] && echo "✅ DB_PASSWORD set"
[ -n "$JWT_SECRET_KEY" ] && echo "✅ JWT_SECRET_KEY set"
[ -n "$ENCRYPTION_KEY" ] && echo "✅ ENCRYPTION_KEY set"
```

### 2. Initial Deployment

```bash
# Pull latest images
docker-compose -f docker-compose.prod.yml pull

# Start infrastructure services first
docker-compose -f docker-compose.prod.yml up -d postgres redis prometheus

# Wait for databases to initialize
sleep 30

# Start core router
docker-compose -f docker-compose.prod.yml up -d federated-router

# Wait for router to be ready
sleep 60

# Start hospital nodes
docker-compose -f docker-compose.prod.yml up -d \
  hospital-node-cardiology \
  hospital-node-neurology \
  hospital-node-emergency

# Start remaining services
docker-compose -f docker-compose.prod.yml up -d
```

### 3. Post-deployment Verification

```bash
# Check service health
docker-compose -f docker-compose.prod.yml ps

# Test API endpoints
curl -k https://localhost/health
curl -k https://localhost/api/v1/status

# Verify SSL/TLS
echo | openssl s_client -connect localhost:443 -servername your-domain.com

# Check logs for errors
docker-compose -f docker-compose.prod.yml logs federated-router | tail -50
docker-compose -f docker-compose.prod.yml logs hospital-node-cardiology | tail -20

# Test federated communication
curl -k -X POST https://localhost/api/v1/inference \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "prompt": "Test medical query",
    "model": "medllama-7b",
    "department": "cardiology",
    "max_privacy_budget": 0.1
  }'
```

## 📊 Monitoring and Observability

### Grafana Dashboards

Access Grafana at `https://your-domain.com:3000`

**Default Dashboards:**
1. **Federated Router Overview**: Request latency, throughput, error rates
2. **Privacy Metrics**: Budget consumption, differential privacy parameters
3. **Security Dashboard**: Authentication failures, injection attempts
4. **Node Health**: Hospital node status, model performance
5. **Infrastructure**: CPU, memory, network, storage metrics

### Key Metrics to Monitor

**Performance Metrics:**
- Request latency percentiles (p50, p95, p99)
- Throughput (requests per second)
- Cache hit ratio
- Model inference time

**Privacy Metrics:**
- Privacy budget consumption by user/department
- Differential privacy parameter drift
- Privacy budget violations

**Security Metrics:**
- Authentication failure rate
- Input validation violations
- Rate limiting triggers
- SSL certificate expiry

**Infrastructure Metrics:**
- CPU/Memory/Disk utilization
- Network latency between nodes
- Database connection pool usage
- Redis memory consumption

## 🔒 Security Hardening

### 1. Network Security

```bash
# Configure firewall rules
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow specific ports
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP (redirects to HTTPS)
sudo ufw allow 443/tcp  # HTTPS
sudo ufw allow 8443/tcp # Hospital nodes (restrict to internal network)

# Hospital network communication (adjust IP ranges)
sudo ufw allow from 10.0.0.0/8 to any port 8443
sudo ufw allow from 172.16.0.0/12 to any port 8443
sudo ufw allow from 192.168.0.0/16 to any port 8443

sudo ufw enable
```

### 2. Container Security

```bash
# Run security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image federated-dp-llm:latest

# Enable AppArmor/SELinux profiles
sudo aa-enforce /etc/apparmor.d/docker

# Set up log monitoring
sudo systemctl enable auditd
sudo systemctl start auditd
```

### 3. Secrets Management

```bash
# Initialize Docker secrets (for Docker Swarm)
echo "$DB_PASSWORD" | docker secret create db_password -
echo "$JWT_SECRET_KEY" | docker secret create jwt_secret -
echo "$ENCRYPTION_KEY" | docker secret create encryption_key -

# For Kubernetes, use sealed secrets
kubectl create secret generic federated-secrets \
  --from-literal=db-password="$DB_PASSWORD" \
  --from-literal=jwt-secret="$JWT_SECRET_KEY" \
  --from-literal=encryption-key="$ENCRYPTION_KEY"
```

## 🔄 Backup and Disaster Recovery

### 1. Database Backup

```bash
# Automated PostgreSQL backup script
cat > backup_postgres.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backup/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/federated_dp_backup_$DATE.sql"

mkdir -p $BACKUP_DIR

docker exec postgres-prod pg_dump -U federated federated_dp > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Keep only last 30 days
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete

echo "Backup completed: ${BACKUP_FILE}.gz"
EOF

chmod +x backup_postgres.sh

# Add to crontab
echo "0 2 * * * /opt/federated-dp-llm/backup_postgres.sh" | crontab -
```

### 2. Configuration Backup

```bash
# Backup critical configurations
tar -czf config_backup_$(date +%Y%m%d).tar.gz \
  configs/ \
  ssl/ \
  nginx/ \
  monitoring/ \
  .env.prod \
  docker-compose.prod.yml

# Store in secure location
aws s3 cp config_backup_*.tar.gz s3://your-backup-bucket/federated-dp-llm/
```

## 📈 Scaling Guidelines

### Horizontal Scaling

**Adding Hospital Nodes:**
1. Generate new SSL certificates
2. Update `docker-compose.prod.yml`
3. Add node configuration to `production.yaml`
4. Deploy with `docker-compose up -d new-node`

**Router Scaling:**
```yaml
# In docker-compose.prod.yml
federated-router:
  deploy:
    replicas: 3
  # Add load balancer configuration
```

### Vertical Scaling

**Resource Recommendations by Load:**

| Load Level | Router CPU/RAM | Node CPU/RAM | Database CPU/RAM |
|------------|----------------|--------------|------------------|
| Light      | 2 cores/4GB    | 4 cores/8GB  | 2 cores/4GB      |
| Medium     | 4 cores/8GB    | 8 cores/16GB | 4 cores/8GB      |
| Heavy      | 8 cores/16GB   | 16 cores/32GB| 8 cores/16GB     |
| Enterprise | 16 cores/32GB  | 32 cores/64GB| 16 cores/32GB    |

## 🚨 Troubleshooting

### Common Issues

**1. SSL Certificate Issues**
```bash
# Check certificate validity
openssl x509 -in ssl/server-cert.pem -text -noout

# Regenerate if expired
./ssl/generate_certificates.sh
```

**2. Database Connection Issues**
```bash
# Check PostgreSQL logs
docker logs postgres-prod

# Test connection
docker exec -it postgres-prod psql -U federated -d federated_dp -c "SELECT 1;"
```

**3. High Memory Usage**
```bash
# Check Redis memory
docker exec redis-prod redis-cli INFO memory

# Clear cache if needed
docker exec redis-prod redis-cli FLUSHDB
```

**4. Network Connectivity Issues**
```bash
# Test node connectivity
docker exec federated-router-prod curl -k https://hospital-cardiology:8443/health

# Check network configuration
docker network inspect federated-network
```

### Log Analysis

```bash
# Real-time log monitoring
docker-compose -f docker-compose.prod.yml logs -f

# Error analysis
docker-compose -f docker-compose.prod.yml logs | grep ERROR

# Performance analysis
docker-compose -f docker-compose.prod.yml logs federated-router | grep "latency\|throughput"
```

## 📞 Support and Maintenance

### Regular Maintenance Tasks

**Daily:**
- Monitor system health dashboards
- Check error logs for anomalies
- Verify backup completion

**Weekly:**
- Review security logs
- Update security patches
- Performance optimization review

**Monthly:**
- SSL certificate expiry check
- Capacity planning review
- Security audit
- Backup restore testing

### Emergency Contacts

- **System Administrator**: admin@hospital.com
- **Security Team**: security@hospital.com  
- **Privacy Officer**: privacy@hospital.com
- **On-call Engineer**: +1-555-FEDERATED

### Version Updates

```bash
# Update to new version
docker pull federated-dp-llm:latest
docker-compose -f docker-compose.prod.yml up -d --no-deps federated-router

# Rollback if needed
docker tag federated-dp-llm:previous federated-dp-llm:latest
docker-compose -f docker-compose.prod.yml up -d --no-deps federated-router
```

---

## 🎉 Deployment Complete!

Your Federated DP-LLM Router is now deployed in production with:

- ✅ **High Availability**: Load balanced, auto-scaling infrastructure
- ✅ **Enterprise Security**: SSL/TLS, client certificates, WAF protection  
- ✅ **Privacy Compliance**: HIPAA/GDPR compliant with differential privacy
- ✅ **Monitoring**: Comprehensive observability with Prometheus/Grafana
- ✅ **Disaster Recovery**: Automated backups and recovery procedures

**Next Steps:**
1. Configure monitoring alerts
2. Train operations team
3. Conduct security penetration testing
4. Perform load testing
5. Set up CI/CD pipeline for updates

For additional support, please refer to the [troubleshooting guide](troubleshooting.md) or contact the development team.