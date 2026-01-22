#!/bin/bash

echo "🚀 Starting CKD Detection System..."

# Navigate to docker directory
cd docker

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build and start services
echo "🔨 Building and starting services..."
docker-compose up -d --build --force-recreate

# Wait for services
echo "⏳ Waiting for services to be healthy..."
sleep 10

# Check status
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "✅ System Started!"
echo ""
echo "📍 Access Points:"
echo "   Internal Dashboard: http://localhost:8501"
echo "   User App:           http://localhost:8502"
echo "   Backend API:        http://localhost:8000/docs"
echo "   MLflow UI:          http://localhost:5050"
echo ""
echo "🔧 Useful Commands:"
echo "   View logs:          docker-compose logs -f"
echo "   Stop system:        docker-compose down"
echo "   Restart backend:    docker-compose restart backend"
echo ""