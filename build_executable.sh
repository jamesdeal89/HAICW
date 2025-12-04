#!/bin/bash
# Build standalone executable for BlackSmith's Bookstore Chatbot

echo "Building standalone executable..."

# Install PyInstaller if not already installed
pip install pyinstaller

# Build the executable
pyinstaller --onefile \
    --name bookstore-chatbot \
    --add-data "intents.csv:." \
    --add-data "qa.csv:." \
    --add-data "stock.json:." \
    --add-data "locations.json:." \
    --add-data "orders.json:." \
    --add-data "session.json:." \
    --add-data "feedback.json:." \
    --hidden-import=nltk \
    --hidden-import=sklearn.feature_extraction.text \
    --hidden-import=numpy \
    --collect-data nltk \
    main.py

echo "Build complete! Executable is in ./dist/bookstore-chatbot"
echo "Run it with: ./dist/bookstore-chatbot"
