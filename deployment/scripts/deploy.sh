#!/bin/bash

# File Processing System Deployment Script
# This script deploys the file processing system to Kubernetes

set -e

# Configuration
NAMESPACE="file-processing"
IMAGE_TAG="${IMAGE_TAG:-latest}"
REGISTRY="${REGISTRY:-localhost:5000}"
IMAGE_NAME="${IMAGE_NAME:-file-processing}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."
    
    docker build -t "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}" .
    
    if [ $? -eq 0 ]; then
        log_info "Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Push Docker image
push_image() {
    log_info "Pushing Docker image to registry..."
    
    docker push "${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"
    
    if [ $? -eq 0 ]; then
        log_info "Docker image pushed successfully"
    else
        log_error "Failed to push Docker image"
        exit 1
    fi
}

# Create namespace
create_namespace() {
    log_info "Creating namespace..."
    
    kubectl apply -f deployment/kubernetes/namespace.yaml
    
    if [ $? -eq 0 ]; then
        log_info "Namespace created/updated successfully"
    else
        log_error "Failed to create namespace"
        exit 1
    fi
}

# Deploy ConfigMap
deploy_configmap() {
    log_info "Deploying ConfigMap..."
    
    kubectl apply -f deployment/kubernetes/configmap.yaml
    
    if [ $? -eq 0 ]; then
        log_info "ConfigMap deployed successfully"
    else
        log_error "Failed to deploy ConfigMap"
        exit 1
    fi
}

# Deploy Redis
deploy_redis() {
    log_info "Deploying Redis..."
    
    kubectl apply -f deployment/kubernetes/redis-deployment.yaml
    
    if [ $? -eq 0 ]; then
        log_info "Redis deployed successfully"
    else
        log_error "Failed to deploy Redis"
        exit 1
    fi
    
    # Wait for Redis to be ready
    log_info "Waiting for Redis to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/redis -n ${NAMESPACE}
}

# Deploy web application
deploy_web() {
    log_info "Deploying web application..."
    
    # Update image in deployment
    sed -i "s|image: file-processing:latest|image: ${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}|g" deployment/kubernetes/web-deployment.yaml
    
    kubectl apply -f deployment/kubernetes/web-deployment.yaml
    
    if [ $? -eq 0 ]; then
        log_info "Web application deployed successfully"
    else
        log_error "Failed to deploy web application"
        exit 1
    fi
    
    # Wait for deployment to be ready
    log_info "Waiting for web application to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/file-processing-web -n ${NAMESPACE}
}

# Deploy HPA
deploy_hpa() {
    log_info "Deploying Horizontal Pod Autoscaler..."
    
    kubectl apply -f deployment/kubernetes/hpa.yaml
    
    if [ $? -eq 0 ]; then
        log_info "HPA deployed successfully"
    else
        log_error "Failed to deploy HPA"
        exit 1
    fi
}

# Deploy Ingress
deploy_ingress() {
    log_info "Deploying Ingress..."
    
    kubectl apply -f deployment/kubernetes/ingress.yaml
    
    if [ $? -eq 0 ]; then
        log_info "Ingress deployed successfully"
    else
        log_error "Failed to deploy Ingress"
        exit 1
    fi
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=file-processing-web -n ${NAMESPACE} --timeout=300s
    
    # Get service endpoint
    SERVICE_IP=$(kubectl get service file-processing-web-service -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
    
    # Port forward for testing
    kubectl port-forward service/file-processing-web-service 8080:80 -n ${NAMESPACE} &
    PORT_FORWARD_PID=$!
    
    sleep 5
    
    # Test health endpoint
    if curl -f http://localhost:8080/health &> /dev/null; then
        log_info "Health check passed"
    else
        log_error "Health check failed"
        kill $PORT_FORWARD_PID
        exit 1
    fi
    
    kill $PORT_FORWARD_PID
}

# Display deployment status
show_status() {
    log_info "Deployment Status:"
    echo "===================="
    
    kubectl get pods -n ${NAMESPACE}
    echo ""
    kubectl get services -n ${NAMESPACE}
    echo ""
    kubectl get ingress -n ${NAMESPACE}
    echo ""
    kubectl get hpa -n ${NAMESPACE}
}

# Main deployment function
main() {
    log_info "Starting deployment of File Processing System"
    
    check_prerequisites
    build_image
    
    if [ "${PUSH_IMAGE}" = "true" ]; then
        push_image
    fi
    
    create_namespace
    deploy_configmap
    deploy_redis
    deploy_web
    deploy_hpa
    deploy_ingress
    health_check
    show_status
    
    log_info "Deployment completed successfully!"
    log_info "Access the application at: http://file-processing.example.com"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH_IMAGE="true"
            shift
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --push          Push image to registry"
            echo "  --tag TAG       Docker image tag (default: latest)"
            echo "  --registry REG  Docker registry (default: localhost:5000)"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main