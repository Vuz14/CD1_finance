#!/bin/bash
# Script to run entire application

echo "🚀 Starting ML Model Trainer Application"

# Check if node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Create results directory
mkdir -p backend/results
mkdir -p backend/uploads

# Start backend
echo "🔧 Starting backend server..."
cd backend
python app.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
echo "🎨 Starting frontend server..."
cd ../frontend
npm start &
FRONTEND_PID=$!

# Wait for services
echo "⏳ Waiting for services to start..."
echo "📍 Backend: http://localhost:5000"
echo "📍 Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both servers"

# Handle cleanup
trap "kill $BACKEND_PID $FRONTEND_PID" INT

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
