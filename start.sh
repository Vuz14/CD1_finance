#!/bin/bash
set -e

echo "Starting CD1 Finance"

mkdir -p backend/uploads backend/results

if [ ! -x ".venv/Scripts/python.exe" ] && [ ! -x ".venv/bin/python" ]; then
  echo "Creating Python virtual environment..."
  python -m venv .venv
fi

if [ -x ".venv/Scripts/python.exe" ]; then
  PYTHON=".venv/Scripts/python.exe"
else
  PYTHON=".venv/bin/python"
fi

echo "Installing backend dependencies into .venv..."
$PYTHON -m pip install -r backend/requirements.txt

if [ ! -d "frontend/node_modules" ]; then
  echo "Installing frontend dependencies..."
  (cd frontend && npm install)
fi

echo "Starting backend at http://localhost:5000"
(cd backend && ../$PYTHON -m scripts.serve) &
BACKEND_PID=$!

sleep 3

echo "Starting frontend at http://localhost:3000"
(cd frontend && npm start) &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID" INT TERM
wait $BACKEND_PID $FRONTEND_PID
