# Production Deployment Guide

## Prerequisites

### System Requirements
- Docker Engine 20.10+
- Docker Compose 2.0+
- Minimum 8GB RAM
- 50GB+ storage space
- SSL certificates for HTTPS

### Security Requirements
- Generate strong encryption keys
- Configure SSL/TLS certificates
- Set up firewall rules
- Configure network security

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd federated-dp-llm-router
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.production.example .env.production

# Edit with your secure values
nano .env.production
```

**CRITICAL: Set these environment variables:**
```bash
# Generate a secure JWT secret (minimum 32 characters)
JWT_SECRET_KEY=$(openssl rand -base64 32)

# Generate master encryption key
HIPAA_MASTER_KEY=$(openssl rand -base64 64)

# Set strong database passwords
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)
GRAFANA_PASSWORD=$(openssl rand -base64 32)
```

### 3. SSL Certificate Setup

```bash
# Create SSL directory
mkdir -p ssl

# Generate self-signed certificates (for testing)
openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes

# For production, use certificates from a trusted CA
# Place your certificates in:
# - ssl/cert.pem (certificate)
# - ssl/key.pem (private key)
# - ssl/ca.pem (certificate authority)
```

### 4. Deploy Services

```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# Check service status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f federated-router
```

### 5. Initialize System

```bash
# Run quality gates to verify deployment
docker exec federated-dp-llm-router python quality_gates.py

# Check health endpoints
curl -k https://localhost:8443/health
```

## Service URLs

- **Main API**: https://localhost:8443
- **Health Check**: https://localhost:8443/health
- **API Documentation**: https://localhost:8443/docs
- **Grafana Dashboard**: http://localhost:3000
- **Prometheus Metrics**: http://localhost:9090
- **Kibana Logs**: http://localhost:5601

## Security Configuration

### 1. Firewall Rules

```bash
# Allow only necessary ports
sudo ufw allow 22/tcp          # SSH
sudo ufw allow 80/tcp          # HTTP (redirect to HTTPS)
sudo ufw allow 443/tcp         # HTTPS
sudo ufw allow 8443/tcp        # Application HTTPS
sudo ufw enable
```

### 2. Network Security

```yaml
# Update docker-compose.prod.yml networks as needed
networks:
  federated-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### 3. Access Control

```bash
# Create admin user
docker exec -it federated-dp-llm-router python -c "
from federated_dp_llm.security.access_control import get_access_control_manager, Role
manager = get_access_control_manager()
user, msg = manager.create_user(
    username='admin',
    email='admin@yourorganization.com',
    password='YourSecurePassword123!',
    department='administration',
    roles=[Role.SYSTEM_ADMIN]
)
print(f'Admin user created: {msg}')
"
```

## Monitoring and Maintenance

### 1. Health Monitoring

```bash
# Check system health
curl -k https://localhost:8443/health

# Monitor container health
docker-compose -f docker-compose.prod.yml ps

# View resource usage
docker stats
```

### 2. Log Management

```bash
# View application logs
docker-compose -f docker-compose.prod.yml logs federated-router

# View audit logs
docker exec federated-dp-llm-router ls -la /app/logs/

# Rotate logs (setup cron job)
docker exec federated-dp-llm-router logrotate /etc/logrotate.conf
```

### 3. Backup Procedures

```bash
# Database backup
docker exec federated-postgres pg_dump -U feduser federated_dp_llm > backup_$(date +%Y%m%d).sql

# Volume backup
docker run --rm -v federated-dp-llm_postgres-data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup_$(date +%Y%m%d).tar.gz /data
```

### 4. Security Scanning

```bash
# Run container vulnerability scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image federated-dp-llm-router:latest

# Run quality gates
docker exec federated-dp-llm-router python quality_gates.py
```

## Compliance and Auditing

### 1. HIPAA Compliance Checklist

- [x] Encryption at rest and in transit
- [x] Access controls and authentication
- [x] Audit logging enabled
- [x] Data minimization implemented
- [x] Backup and recovery procedures
- [x] Security incident response plan

### 2. Audit Log Retention

```bash
# Check audit log retention
docker exec federated-postgres psql -U feduser -d federated_dp_llm -c "
SELECT 
    COUNT(*) as total_events,
    MIN(timestamp) as oldest_event,
    MAX(timestamp) as newest_event
FROM audit_events;
"
```

## Troubleshooting

### Common Issues

1. **SSL Certificate Errors**
   ```bash
   # Check certificate validity
   openssl x509 -in ssl/cert.pem -text -noout
   
   # Verify certificate chain
   openssl verify -CAfile ssl/ca.pem ssl/cert.pem
   ```

2. **Database Connection Issues**
   ```bash
   # Check database connectivity
   docker exec federated-postgres pg_isready -U feduser
   
   # Check database logs
   docker-compose -f docker-compose.prod.yml logs postgres
   ```

3. **Memory Issues**
   ```bash
   # Check memory usage
   docker stats --no-stream
   
   # Adjust memory limits in docker-compose.prod.yml
   deploy:
     resources:
       limits:
         memory: 8G
   ```

### Performance Optimization

1. **Scale Services**
   ```bash
   # Scale application instances
   docker-compose -f docker-compose.prod.yml up -d --scale federated-router=3
   ```

2. **Database Optimization**
   ```sql
   -- Connect to database and optimize
   docker exec -it federated-postgres psql -U feduser -d federated_dp_llm
   
   -- Create indexes for better performance
   CREATE INDEX idx_audit_timestamp ON audit_events(timestamp);
   CREATE INDEX idx_privacy_user_id ON privacy_budget(user_id);
   ```

## Updates and Maintenance

### 1. Application Updates

```bash
# Pull latest changes
git pull origin main

# Rebuild and deploy
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Run quality gates after update
docker exec federated-dp-llm-router python quality_gates.py
```

### 2. Security Updates

```bash
# Update base images
docker-compose -f docker-compose.prod.yml pull

# Update system packages
docker-compose -f docker-compose.prod.yml build --no-cache
```

## Support and Documentation

- **API Documentation**: https://localhost:8443/docs
- **System Logs**: `/app/logs/`
- **Configuration**: `/app/config/production.yaml`
- **SSL Certificates**: `/app/ssl/`

For additional support, check the monitoring dashboards and audit logs for detailed system information.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Configuration](#configuration)
6. [Security Setup](#security-setup)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Scaling and Performance](#scaling-and-performance)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance](#maintenance)

## Overview

The Federated DP-LLM system is designed for high-availability, secure deployment in healthcare environments with strict compliance requirements. The system supports multiple deployment options:

- **Docker Compose**: For development and small-scale deployments
- **Kubernetes**: For production-scale deployments with auto-scaling
- **Hybrid Cloud**: For multi-region federated deployments

### Architecture Components

- **Federated Router**: Main coordination service
- **Hospital Nodes**: Federated learning participants
- **Redis**: Caching and session storage
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards
- **Nginx**: Load balancing and SSL termination

## Prerequisites

### System Requirements

**Minimum Hardware:**
- CPU: 8 cores (16 threads)
- RAM: 16 GB
- Storage: 500 GB SSD
- Network: 1 Gbps

**Recommended Hardware:**
- CPU: 16 cores (32 threads)
- RAM: 64 GB
- Storage: 2 TB NVMe SSD
- Network: 10 Gbps
- GPU: NVIDIA A100 (optional, for local inference)

### Software Dependencies

- Docker 24.0+ with Docker Compose
- Kubernetes 1.28+ (for K8s deployment)
- OpenSSL for certificate generation
- kubectl and helm (for K8s management)

### Network Requirements

- **Inbound Ports:**
  - 80/443: HTTP/HTTPS traffic
  - 8080: Main application API
  - 8090: Prometheus metrics
  - 9090: Prometheus UI
  - 3000: Grafana UI

- **Outbound Ports:**
  - 443: HTTPS to external services
  - 6379: Redis (internal)
  - 8443: Hospital node communication

## Docker Deployment

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/federated-dp-llm.git
   cd federated-dp-llm
   ```

2. **Generate SSL certificates:**
   ```bash
   ./scripts/generate_certs.sh
   ```

3. **Configure environment variables:**
   ```bash
   cp configs/docker.env.example configs/docker.env
   # Edit configs/docker.env with your settings
   ```

4. **Start the system:**
   ```bash
   docker-compose up -d
   ```

5. **Verify deployment:**
   ```bash
   curl -k https://localhost/health
   ```

### Production Docker Setup

1. **Build production images:**
   ```bash
   docker build -t federated-dp-llm:latest .
   docker build -f Dockerfile.node -t federated-node:latest .
   ```

2. **Configure production environment:**
   ```bash
   # Set production environment variables
   export ENVIRONMENT=production
   export JWT_SECRET_KEY="$(openssl rand -hex 32)"
   export DATABASE_URL="postgresql://user:pass@db:5432/federated_dp"
   export REDIS_PASSWORD="$(openssl rand -hex 16)"
   ```

3. **Deploy with production compose file:**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

### Docker Security Hardening

1. **Enable Docker Content Trust:**
   ```bash
   export DOCKER_CONTENT_TRUST=1
   ```

2. **Configure Docker daemon security:**
   ```json
   {
     "icc": false,
     "userland-proxy": false,
     "no-new-privileges": true,
     "seccomp-profile": "/etc/docker/seccomp-profile.json"
   }
   ```

3. **Use security scanning:**
   ```bash
   docker scan federated-dp-llm:latest
   ```

## Kubernetes Deployment

### Prerequisites

1. **Install kubectl and helm:**
   ```bash
   # Install kubectl
   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
   sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
   
   # Install helm
   curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
   ```

2. **Verify cluster access:**
   ```bash
   kubectl cluster-info
   kubectl get nodes
   ```

### Deployment Steps

1. **Create namespace and RBAC:**
   ```bash
   kubectl apply -f deployment/kubernetes/namespace.yaml
   kubectl apply -f deployment/kubernetes/rbac.yaml
   ```

2. **Deploy secrets and config maps:**
   ```bash
   kubectl apply -f deployment/kubernetes/secrets.yaml
   kubectl apply -f deployment/kubernetes/configmaps.yaml
   ```

3. **Deploy persistent volumes:**
   ```bash
   kubectl apply -f deployment/kubernetes/pv.yaml
   ```

4. **Deploy Redis and monitoring:**
   ```bash
   kubectl apply -f deployment/kubernetes/redis.yaml
   kubectl apply -f deployment/kubernetes/prometheus.yaml
   kubectl apply -f deployment/kubernetes/grafana.yaml
   ```

5. **Deploy the main application:**
   ```bash
   kubectl apply -f deployment/kubernetes/federated-router-deployment.yaml
   ```

6. **Deploy ingress controller:**
   ```bash
   kubectl apply -f deployment/kubernetes/ingress.yaml
   ```

### Verification

```bash
# Check deployment status
kubectl get pods -n federated-dp-llm
kubectl get services -n federated-dp-llm

# Check application health
kubectl port-forward -n federated-dp-llm svc/federated-router-service 8080:80
curl http://localhost:8080/health
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment federated-router -n federated-dp-llm --replicas=5

# Enable horizontal pod autoscaler
kubectl apply -f deployment/kubernetes/hpa.yaml
```

## Configuration

### Environment Variables

Create configuration files based on your environment:

```yaml
# configs/production.yaml
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4

privacy:
  epsilon_per_query: 0.1
  max_budget_per_user: 10.0
  enable_audit: true

security:
  jwt_secret_key: "${JWT_SECRET_KEY}"
  enable_mtls: true
  require_client_cert: true

database:
  url: "${DATABASE_URL}"
  pool_size: 20

redis:
  url: "${REDIS_URL}"
  max_connections: 50
```

### Department Budget Configuration

Configure privacy budgets for different hospital departments:

```yaml
compliance:
  department_budgets:
    emergency: 20.0      # Higher budget for critical care
    cardiology: 15.0     # Moderate budget for specialized care
    radiology: 10.0      # Standard budget for imaging
    general: 5.0         # Lower budget for general queries
    research: 2.0        # Minimal budget for research use
```

### Network Configuration

Configure hospital node endpoints:

```yaml
federation:
  nodes:
    - id: "hospital_mayo"
      endpoint: "https://mayo-node.hospital.org:8443"
      department: "cardiology"
      region: "us-east-1"
    - id: "hospital_cleveland"
      endpoint: "https://cleveland-node.hospital.org:8443"
      department: "emergency"
      region: "us-east-1"
```

## Security Setup

### Certificate Management

1. **Generate CA certificate:**
   ```bash
   openssl genrsa -out ca.key 4096
   openssl req -new -x509 -days 3650 -key ca.key -out ca.crt \
     -subj "/C=US/ST=CA/L=SF/O=Hospital/OU=IT/CN=Federated-DP-LLM-CA"
   ```

2. **Generate server certificates:**
   ```bash
   openssl genrsa -out server.key 4096
   openssl req -new -key server.key -out server.csr \
     -subj "/C=US/ST=CA/L=SF/O=Hospital/OU=IT/CN=federated-dp-llm.hospital.org"
   openssl x509 -req -days 365 -in server.csr -CA ca.crt -CAkey ca.key \
     -CAcreateserial -out server.crt
   ```

3. **Generate client certificates for each hospital:**
   ```bash
   ./scripts/generate_client_cert.sh hospital_mayo cardiology
   ./scripts/generate_client_cert.sh hospital_cleveland emergency
   ```

### JWT Configuration

```bash
# Generate secure JWT secret
export JWT_SECRET_KEY="$(openssl rand -hex 64)"

# Configure JWT settings
export JWT_ALGORITHM="HS256"
export JWT_EXPIRATION_HOURS="24"
```

### Database Security

```bash
# Create database with encryption
createdb -E UTF8 -T template1 federated_dp

# Configure SSL connections
export DATABASE_URL="postgresql://user:pass@localhost:5432/federated_dp?sslmode=require"
```

## Monitoring and Logging

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'federated-router'
    static_configs:
      - targets: ['federated-router:8090']
    scrape_interval: 10s
    metrics_path: /metrics

  - job_name: 'hospital-nodes'
    static_configs:
      - targets: ['hospital-node-1:8443', 'hospital-node-2:8443']
```

### Grafana Dashboards

Import the provided dashboards:

1. **System Metrics Dashboard:** `deployment/monitoring/dashboards/system-metrics.json`
2. **Privacy Metrics Dashboard:** `deployment/monitoring/dashboards/privacy-metrics.json`
3. **Federation Dashboard:** `deployment/monitoring/dashboards/federation-metrics.json`

### Log Aggregation

Configure centralized logging with ELK stack:

```yaml
# logstash.yml
input:
  beats:
    port: 5044
    
filter:
  if [fields][service] == "federated-router" {
    json {
      source => "message"
    }
  }

output:
  elasticsearch:
    hosts: ["elasticsearch:9200"]
    index => "federated-dp-llm-%{+YYYY.MM.dd}"
```

### Alerting Rules

```yaml
# alerting-rules.yml
groups:
  - name: federated-dp-llm
    rules:
      - alert: HighPrivacyBudgetUsage
        expr: privacy_budget_remaining < 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Privacy budget running low for user {{ $labels.user_id }}"
          
      - alert: NodeUnhealthy
        expr: up{job="hospital-nodes"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Hospital node {{ $labels.instance }} is down"
```

## Scaling and Performance

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: federated-router-hpa
  namespace: federated-dp-llm
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: federated-router
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Performance Tuning

1. **Database Connection Pooling:**
   ```yaml
   database:
     pool_size: 20
     max_overflow: 30
     pool_timeout: 30
     pool_recycle: 3600
   ```

2. **Redis Optimization:**
   ```yaml
   redis:
     max_connections: 100
     socket_timeout: 5
     health_check_interval: 30
     maxmemory_policy: "allkeys-lru"
   ```

3. **Nginx Optimization:**
   ```nginx
   worker_processes auto;
   worker_connections 4096;
   keepalive_timeout 65;
   client_max_body_size 10M;
   ```

### Load Testing

```bash
# Install k6
curl https://github.com/grafana/k6/releases/download/v0.45.0/k6-v0.45.0-linux-amd64.tar.gz -L | tar xvz --strip-components 1

# Run load test
k6 run deployment/tests/load-test.js
```

## Troubleshooting

### Common Issues

1. **Certificate Issues:**
   ```bash
   # Verify certificate chain
   openssl verify -CAfile ca.crt server.crt
   
   # Check certificate expiration
   openssl x509 -in server.crt -noout -dates
   ```

2. **Database Connection Issues:**
   ```bash
   # Test database connectivity
   kubectl exec -it federated-router-0 -- psql $DATABASE_URL -c "SELECT 1"
   ```

3. **Redis Connection Issues:**
   ```bash
   # Test Redis connectivity
   kubectl exec -it redis-0 -- redis-cli ping
   ```

### Log Analysis

```bash
# View application logs
kubectl logs -f deployment/federated-router -n federated-dp-llm

# View audit logs
kubectl logs -f deployment/federated-router -n federated-dp-llm -c audit-logger

# Search for errors
kubectl logs deployment/federated-router -n federated-dp-llm | grep ERROR
```

### Performance Debugging

```bash
# Check resource usage
kubectl top pods -n federated-dp-llm

# Profile application
kubectl exec -it federated-router-0 -- python -m cProfile -o profile.stats app.py
```

## Maintenance

### Backup Procedures

1. **Database Backup:**
   ```bash
   kubectl exec postgres-0 -- pg_dump federated_dp > backup-$(date +%Y%m%d).sql
   ```

2. **Redis Backup:**
   ```bash
   kubectl exec redis-0 -- redis-cli SAVE
   kubectl cp redis-0:/data/dump.rdb ./redis-backup-$(date +%Y%m%d).rdb
   ```

3. **Configuration Backup:**
   ```bash
   kubectl get configmaps -n federated-dp-llm -o yaml > configmaps-backup.yaml
   kubectl get secrets -n federated-dp-llm -o yaml > secrets-backup.yaml
   ```

### Update Procedures

1. **Rolling Update:**
   ```bash
   kubectl set image deployment/federated-router \
     federated-router=federated-dp-llm:v1.1.0 -n federated-dp-llm
   ```

2. **Database Migration:**
   ```bash
   kubectl exec -it federated-router-0 -- python -m federated_dp_llm.cli migrate
   ```

3. **Configuration Updates:**
   ```bash
   kubectl apply -f deployment/kubernetes/configmaps.yaml
   kubectl rollout restart deployment/federated-router -n federated-dp-llm
   ```

### Health Monitoring

```bash
# Check system health
curl -s https://federated-dp-llm.hospital.org/health | jq .

# Monitor privacy budgets
curl -s https://federated-dp-llm.hospital.org/api/privacy/budgets | jq .

# Check federation status
curl -s https://federated-dp-llm.hospital.org/api/federation/status | jq .
```

### Security Auditing

```bash
# Review audit logs
kubectl logs -l app=federated-router --since=24h | grep AUDIT

# Check certificate expiration
openssl x509 -in /etc/ssl/certs/server.crt -noout -checkend 604800

# Verify compliance
python -m federated_dp_llm.cli compliance-check --report
```

## Support and Contact

For deployment support and issues:

- **Documentation:** https://docs.federated-dp-llm.org
- **GitHub Issues:** https://github.com/your-org/federated-dp-llm/issues
- **Security Issues:** security@hospital.org
- **Support Email:** support@federated-dp-llm.org

## License and Compliance

This deployment guide assumes compliance with:
- HIPAA (Health Insurance Portability and Accountability Act)
- GDPR (General Data Protection Regulation)
- CCPA (California Consumer Privacy Act)
- SOC 2 Type II certification requirements

Ensure all deployment configurations meet your organization's specific compliance requirements.