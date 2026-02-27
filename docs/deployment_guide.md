# Deployment Guide

This guide covers deploying the Intelligent Support Ticket Classification system using Docker and cloud platforms.

## Table of Contents

1. [Docker Deployment](#docker-deployment)
2. [Azure Deployment](#azure-deployment)
3. [Environment Setup](#environment-setup)
4. [Monitoring](#monitoring)
5. [Troubleshooting](#troubleshooting)

## Docker Deployment

### Prerequisites

- Docker installed (version 20.10+)
- Docker Compose installed (version 1.29+)

### Building the Docker Image

```bash
# Build the image
docker build -t intelligent-support-rag:latest .

# Run the container
docker run -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e MODEL_NAME=bert-base-uncased \
  intelligent-support-rag:latest
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f api
```

## Azure Deployment

### Prerequisites

- Azure CLI installed
- Azure subscription
- AkS (Azure Kubernetes Service) cluster (optional)

### Deployment Steps

1. **Container Registry Setup**

```bash
# Create Azure Container Registry
az acr create --resource-group myResourceGroup \
  --name myACR --sku Basic

# Login to registry
az acr login --name myACR
```

2. **Push Docker Image**

```bash
# Tag image
docker tag intelligent-support-rag:latest \
  myACR.azurecr.io/intelligent-support-rag:latest

# Push to registry
docker push myACR.azurecr.io/intelligent-support-rag:latest
```

3. **Deploy to App Service**

```bash
# Create App Service Plan
az appservice plan create --name myPlan \
  --resource-group myResourceGroup --sku B2 --is-linux

# Create Web App
az webapp create --resource-group myResourceGroup \
  --plan myPlan --name myWebApp \
  --deployment-container-image-name-user myACR.azurecr.io/intelligent-support-rag:latest
```

## Environment Setup

### Configuration Files

Create a `.env` file with the following variables:

```bash
ENVIRONMENT=production
DEBUG=False
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
MLFLOW_TRACKING_URI=http://mlflow:5000
DATABASE_URL=postgresql://user:password@db:5432/tickets
```

### Database Setup

```bash
# Create database
createdb ticket_database

# Run migrations
python src/deployment/migrations.py
```

## Monitoring

### MLflow Setup

```bash
# Start MLflow server
mlflow server --backend-store-uri postgresql://user@localhost/mlflow \
  --default-artifact-root ./mlruns
```

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Expected response
{"status": "healthy", "message": "Service is running"}
```

### Logging

Logs are stored in `/var/log/app.log`. Configure log rotation using logrotate:

```bash
# Edit logrotate config
sudo nano /etc/logrotate.d/intelligent-support-rag

# Add:
/var/log/app.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 appuser appuser
}
```

## Troubleshooting

### Common Issues

**1. API won't start**

```bash
# Check logs
docker logs intelligent-support-rag

# Verify environment variables
docker exec intelligent-support-rag env | grep ENVIRONMENT

# Test port availability
netstat -tuln | grep 8000
```

**2. Model loading errors**

```bash
# Check model directory
ls -la models/saved/

# Verify model format
python -c "import torch; torch.load('models/saved/model.pt')"
```

**3. Database connection issues**

```bash
# Test connection
psql -h localhost -U user -d ticket_database

# Check connection string
echo $DATABASE_URL
```

### Performance Tuning

- Increase API workers: `API_WORKERS=8`
- Enable caching: `CACHE_ENABLED=True`
- Batch predictions: Set batch size to 32+
- Use GPU if available: `CUDA_VISIBLE_DEVICES=0`

## Support

For issues or questions, contact the development team or check the [GitHub Issues](https://github.com/yourusername/intelligent-support-rag/issues).
