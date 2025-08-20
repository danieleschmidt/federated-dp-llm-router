# Production Deployment Guide

## Overview

This guide covers the production deployment of the Federated DP-LLM Router system for healthcare environments with HIPAA compliance.

## Prerequisites

- Docker and Docker Compose
- Kubernetes cluster (for K8s deployment)
- SSL certificates for HTTPS
- Database (PostgreSQL) and Redis
- Monitoring infrastructure (Prometheus/Grafana)

## Quick Deploy

### Docker Compose (Recommended for single-node)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/federated-dp-llm-router.git
cd federated-dp-llm-router

# 2. Set environment variables
cp .env.example .env
# Edit .env with your configuration

# 3. Deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# 4. Verify deployment
curl http://localhost:8080/health
```

### Kubernetes (Recommended for multi-node)

```bash
# 1. Apply manifests
kubectl apply -f deployment/kubernetes/

# 2. Verify deployment
kubectl get pods -n federated-dp-llm
kubectl get services -n federated-dp-llm

# 3. Check health
kubectl port-forward -n federated-dp-llm service/federated-dp-llm-service 8080:80
curl http://localhost:8080/health
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | production |
| `LOG_LEVEL` | Logging level | INFO |
| `ENABLE_AUDIT_LOGGING` | Enable audit trails | true |
| `HIPAA_COMPLIANCE_MODE` | Enable HIPAA compliance | true |
| `QUANTUM_OPTIMIZATION` | Enable quantum planning | true |

### Secrets Management

Store sensitive configuration in secrets:

```bash
# Kubernetes secrets
kubectl create secret generic federated-dp-llm-secrets \
  --from-literal=postgres-password='your-secure-password' \
  --from-literal=redis-password='your-redis-password' \
  -n federated-dp-llm
```

## Monitoring

### Health Checks

- **Application Health**: `GET /health`
- **Readiness Check**: `GET /ready`
- **Metrics**: `GET /metrics` (Prometheus format)

### Observability Stack

- **Prometheus**: Metrics collection (`http://localhost:9090`)
- **Grafana**: Visualization dashboard (`http://localhost:3000`)
- **Structured Logging**: JSON format with correlation IDs

## Security

### Healthcare Compliance

- HIPAA-compliant logging (PHI redaction)
- Audit trails for all data access
- End-to-end encryption
- Role-based access control

### Network Security

- TLS 1.3 for all communications
- mTLS between services
- Network policies for pod-to-pod communication
- Firewall rules for ingress/egress

## Scaling

### Horizontal Pod Autoscaler (HPA)

Automatically scales based on:
- CPU utilization (70% target)
- Memory utilization (80% target)
- Custom privacy budget metrics

### Manual Scaling

```bash
# Scale to 10 replicas
kubectl scale deployment federated-dp-llm-api --replicas=10 -n federated-dp-llm
```

## Backup and Recovery

### Database Backup

```bash
# Automated daily backups
kubectl create cronjob postgres-backup \
  --image=postgres:15 \
  --schedule="0 2 * * *" \
  --restart=OnFailure \
  -- /bin/sh -c "pg_dump -h postgres -U federated_user federated_dp_llm | gzip > /backup/backup-$(date +%Y%m%d).sql.gz"
```

### Disaster Recovery

- RTO: 15 minutes
- RPO: 1 hour
- Multi-region deployment capability
- Automated failover procedures

## Troubleshooting

### Common Issues

1. **Pod startup failures**: Check resource limits and secrets
2. **Database connection errors**: Verify network policies
3. **High latency**: Check resource allocation and scaling

### Debug Commands

```bash
# Check pod logs
kubectl logs -f deployment/federated-dp-llm-api -n federated-dp-llm

# Exec into pod
kubectl exec -it deployment/federated-dp-llm-api -n federated-dp-llm -- /bin/bash

# Check resource usage
kubectl top pods -n federated-dp-llm
```

## Maintenance

### Updates

1. Update Docker image tag in manifests
2. Apply rolling update: `kubectl rollout restart deployment/federated-dp-llm-api -n federated-dp-llm`
3. Monitor rollout: `kubectl rollout status deployment/federated-dp-llm-api -n federated-dp-llm`

### Health Monitoring

- Set up alerts for critical metrics
- Monitor privacy budget consumption
- Track quantum coherence levels
- Monitor audit log completeness
