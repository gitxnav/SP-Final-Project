#!/bin/bash

echo "🛑 Stopping CKD Detection System..."

# Navigate to docker directory
cd docker

echo "⏳ Stopping all services..."
docker-compose down

# Optional: Remove volumes (commented out by default)
# echo "🗑️  Removing volumes..."
# docker-compose down -v

# Optional: Remove orphaned containers (commented out by default)
# echo "🧹 Removing orphaned containers..."
# docker-compose down --remove-orphans

echo ""
echo "✅ System Stopped Successfully!"
echo ""
echo "📊 Cleanup Commands (if needed):"
echo "   Remove all containers:    docker rm \$(docker ps -aq)"
echo "   Remove all volumes:       docker volume rm \$(docker volume ls -q)"
echo "   Remove all images:        docker rmi \$(docker images -q)"
echo "   Full system cleanup:      docker system prune -a --volumes"
echo ""