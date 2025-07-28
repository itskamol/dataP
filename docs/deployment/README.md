# Deployment Guide

## Overview

This guide covers deployment options for the File Processing Optimization system across different environments, from local development to production Kubernetes clusters.

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 10GB free space
- OS: Linux, macOS, or Windows

**Recommended Requirements:**
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 50GB+ SSD
- OS: Linux (Ubuntu 20.04+ or CentOS 8+)

### Software Dependencies

**Required:**
- Python 3.8+
- pip package manager
- Git

**Optional:**
- Docker 20.10+
- Docker Compose 2.0+
- Kubernetes 1.20+
- Redis 6.0+ (for caching)
- PostgreSQL 13+ (for persistent storage)

## Local Development Setup

### 1. Basic Setup

```bash
# Clone repository
git clone <repository-url>
cd file-processing-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### 2. Configuration

Create `config.json`:
```json
{
  "environment": "development",
  "debug": true,
  "log_level": "DEBUG",
  "file_upload": {
    "max_file_size_mb": 100,
    "allowed_extensions": [".csv", ".json", ".xlsx"],
    "upload_directory": "uploads"
  },
  "processing": {
    "max_workers": 2,
    "memory_limit_mb": 1024,
    "timeout_seconds": 3600
  },
  "cache": {
    "type": "memory",
    "size": 10000,
    "ttl_seconds": 3600
  },
  "web": {
    "host": "127.0.0.1",
    "port": 5000,
    "secret_key": "dev-secret-key-change-in-production"
  }
}
```

### 3. Environment Variables

Create `.env` file:
```bash
# Application
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=dev-secret-key

# Logging
LOG_LEVEL=DEBUG
LOG_FORMAT=json

# Processing
MAX_WORKERS=2
MEMORY_LIMIT_MB=1024

# Cache
CACHE_TYPE=memory
CACHE_SIZE=10000

# Security
JWT_SECRET_KEY=jwt-secret-key
JWT_EXPIRATION_HOURS=24
```

### 4. Running the Application

**Web Interface:**
```bash
# Start web server
python -m src.web.api_app

# Or using Flask CLI
export FLASK_APP=src.web.api_app
flask run --host=0.0.0.0 --port=5000
```

**CLI Tool:**
```bash
# Process files via CLI
python -m src.application.cli --config config.json --file1 data1.csv --file2 data2.json
```

## Docker Deployment

### 1. Single Container

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY setup.py .

# Install application
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Create necessary directories
RUN mkdir -p uploads results logs

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start application
CMD ["python", "-m", "src.web.api_app"]
```

**Build and Run:**
```bash
# Build image
docker build -t file-processor:latest .

# Run container
docker run -d \
  --name file-processor \
  -p 5000:5000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/logs:/app/logs \
  -e FLASK_ENV=production \
  file-processor:latest
```

### 2. Docker Compose

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - REDIS_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://user:password@postgres:5432/fileprocessor
    volumes:
      - ./uploads:/app/uploads
      - ./results:/app/results
      - ./logs:/app/logs
      - ./config:/app/config
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=fileprocessor
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d fileprocessor"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - web
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

**Start Services:**
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f web

# Scale web service
docker-compose up -d --scale web=3

# Stop services
docker-compose down
```

### 3. Production Docker Configuration

**Production Dockerfile:**
```dockerfile
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application
COPY src/ ./src/
COPY config/ ./config/
COPY setup.py .

# Install application
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 appuser
RUN mkdir -p uploads results logs && chown -R appuser:appuser /app
USER appuser

# Security: Remove unnecessary packages
RUN pip uninstall -y pip setuptools

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "300", "src.web.api_app:app"]
```

## Kubernetes Deployment

### 1. Basic Deployment

**namespace.yaml:**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: file-processor
  labels:
    name: file-processor
```

**configmap.yaml:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: file-processor-config
  namespace: file-processor
data:
  config.json: |
    {
      "environment": "production",
      "debug": false,
      "log_level": "INFO",
      "file_upload": {
        "max_file_size_mb": 500,
        "allowed_extensions": [".csv", ".json", ".xlsx"],
        "upload_directory": "/app/uploads"
      },
      "processing": {
        "max_workers": 4,
        "memory_limit_mb": 2048,
        "timeout_seconds": 7200
      },
      "cache": {
        "type": "redis",
        "url": "redis://redis-service:6379/0",
        "ttl_seconds": 3600
      },
      "web": {
        "host": "0.0.0.0",
        "port": 5000
      }
    }
```

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: file-processor
  namespace: file-processor
  labels:
    app: file-processor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: file-processor
  template:
    metadata:
      labels:
        app: file-processor
    spec:
      containers:
      - name: file-processor
        image: file-processor:latest
        ports:
        - containerPort: 5000
        env:
        - name: FLASK_ENV
          value: "production"
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: uploads-volume
          mountPath: /app/uploads
        - name: results-volume
          mountPath: /app/results
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: config-volume
        configMap:
          name: file-processor-config
      - name: uploads-volume
        persistentVolumeClaim:
          claimName: uploads-pvc
      - name: results-volume
        persistentVolumeClaim:
          claimName: results-pvc
```

**service.yaml:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: file-processor-service
  namespace: file-processor
spec:
  selector:
    app: file-processor
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: ClusterIP
```

**ingress.yaml:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: file-processor-ingress
  namespace: file-processor
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: 500m
spec:
  tls:
  - hosts:
    - fileprocessor.example.com
    secretName: file-processor-tls
  rules:
  - host: fileprocessor.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: file-processor-service
            port:
              number: 80
```

### 2. Horizontal Pod Autoscaler

**hpa.yaml:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: file-processor-hpa
  namespace: file-processor
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: file-processor
  minReplicas: 2
  maxReplicas: 10
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

### 3. Persistent Storage

**storage.yaml:**
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: uploads-pvc
  namespace: file-processor
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: results-pvc
  namespace: file-processor
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 200Gi
  storageClassName: fast-ssd
```

### 4. Deploy to Kubernetes

```bash
# Apply all configurations
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f storage.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# Check deployment status
kubectl get pods -n file-processor
kubectl get services -n file-processor
kubectl get ingress -n file-processor

# View logs
kubectl logs -f deployment/file-processor -n file-processor

# Scale deployment
kubectl scale deployment file-processor --replicas=5 -n file-processor
```

## Cloud Deployment

### 1. AWS EKS

**Prerequisites:**
```bash
# Install AWS CLI and eksctl
aws configure
eksctl version
```

**Create EKS Cluster:**
```bash
# Create cluster
eksctl create cluster \
  --name file-processor-cluster \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type m5.large \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --managed

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name file-processor-cluster
```

**Deploy Application:**
```bash
# Deploy to EKS
kubectl apply -f k8s/

# Install ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/aws/deploy.yaml

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml
```

### 2. Google GKE

**Create GKE Cluster:**
```bash
# Set project and zone
gcloud config set project YOUR_PROJECT_ID
gcloud config set compute/zone us-central1-a

# Create cluster
gcloud container clusters create file-processor-cluster \
  --num-nodes=3 \
  --enable-autoscaling \
  --min-nodes=1 \
  --max-nodes=10 \
  --machine-type=n1-standard-2

# Get credentials
gcloud container clusters get-credentials file-processor-cluster
```

### 3. Azure AKS

**Create AKS Cluster:**
```bash
# Create resource group
az group create --name file-processor-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group file-processor-rg \
  --name file-processor-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group file-processor-rg --name file-processor-cluster
```

## Monitoring and Observability

### 1. Prometheus and Grafana

**prometheus.yaml:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'file-processor'
      static_configs:
      - targets: ['file-processor-service.file-processor:80']
      metrics_path: /metrics
      scrape_interval: 30s
```

**Deploy Monitoring Stack:**
```bash
# Install Prometheus Operator
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml

# Deploy Prometheus and Grafana
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace
```

### 2. Logging with ELK Stack

**filebeat-config.yaml:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: filebeat-config
  namespace: logging
data:
  filebeat.yml: |
    filebeat.inputs:
    - type: container
      paths:
        - /var/log/containers/*file-processor*.log
      processors:
      - add_kubernetes_metadata:
          host: ${NODE_NAME}
          matchers:
          - logs_path:
              logs_path: "/var/log/containers/"
    
    output.elasticsearch:
      hosts: ["elasticsearch:9200"]
    
    setup.kibana:
      host: "kibana:5601"
```

## Security Configuration

### 1. Network Policies

**network-policy.yaml:**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: file-processor-netpol
  namespace: file-processor
spec:
  podSelector:
    matchLabels:
      app: file-processor
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 5000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

### 2. Pod Security Policy

**pod-security-policy.yaml:**
```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: file-processor-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

## Backup and Disaster Recovery

### 1. Database Backup

```bash
# PostgreSQL backup
kubectl exec -it postgres-pod -- pg_dump -U user fileprocessor > backup.sql

# Restore
kubectl exec -i postgres-pod -- psql -U user fileprocessor < backup.sql
```

### 2. Volume Backup

```bash
# Create volume snapshot
kubectl apply -f - <<EOF
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: uploads-snapshot
  namespace: file-processor
spec:
  volumeSnapshotClassName: csi-hostpath-snapclass
  source:
    persistentVolumeClaimName: uploads-pvc
EOF
```

## Troubleshooting

### 1. Common Issues

**Pod Not Starting:**
```bash
# Check pod status
kubectl describe pod <pod-name> -n file-processor

# Check logs
kubectl logs <pod-name> -n file-processor

# Check events
kubectl get events -n file-processor --sort-by='.lastTimestamp'
```

**Memory Issues:**
```bash
# Check resource usage
kubectl top pods -n file-processor

# Increase memory limits
kubectl patch deployment file-processor -n file-processor -p '{"spec":{"template":{"spec":{"containers":[{"name":"file-processor","resources":{"limits":{"memory":"4Gi"}}}]}}}}'
```

**Storage Issues:**
```bash
# Check PVC status
kubectl get pvc -n file-processor

# Check storage class
kubectl get storageclass
```

### 2. Performance Tuning

**CPU Optimization:**
```yaml
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"
```

**JVM Tuning (if using Java components):**
```yaml
env:
- name: JAVA_OPTS
  value: "-Xmx2g -Xms1g -XX:+UseG1GC"
```

This deployment guide provides comprehensive instructions for deploying the File Processing Optimization system across various environments, from local development to production Kubernetes clusters with proper monitoring, security, and disaster recovery considerations.