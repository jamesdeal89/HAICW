#!/bin/bash
# Create a portable ZIP distribution

echo "Creating portable distribution package..."

# Create distribution directory
mkdir -p dist_portable
cd dist_portable

# Copy all necessary files
cp ../*.py .
cp ../*.csv .
cp ../*.json .
cp ../*.md .
cp ../requirements.txt .
cp ../LICENSE .

# Create run script
cat > run_chatbot.sh << 'EOF'
#!/bin/bash
# Portable run script for BlackSmith's Bookstore Chatbot

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check if venv exists, create if not
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run the chatbot
python3 main.py

# Deactivate virtual environment
deactivate
EOF

chmod +x run_chatbot.sh

# Create Windows batch file
cat > run_chatbot.bat << 'EOF'
@echo off
REM Portable run script for BlackSmith's Bookstore Chatbot (Windows)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Python 3 is required but not found. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if venv exists, create if not
if not exist "venv\" (
    echo Setting up virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Run the chatbot
python main.py

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat
pause
EOF

# Create README for portable distribution
cat > PORTABLE_README.txt << 'EOF'
BlackSmith's Bookstore Chatbot - Portable Distribution
=======================================================

This is a portable distribution of the chatbot that can run on any system
with Python 3.8 or higher installed.

QUICK START:

  Linux/Mac:
    1. Open terminal in this directory
    2. Run: ./run_chatbot.sh
    
  Windows:
    1. Double-click run_chatbot.bat
    OR
    2. Open Command Prompt in this directory
    3. Run: run_chatbot.bat

FIRST RUN:
  The first time you run the chatbot, it will:
  - Create a virtual environment (venv/)
  - Install required dependencies
  - This takes 1-2 minutes

SUBSEQUENT RUNS:
  The chatbot will start immediately using the existing environment.

MANUAL INSTALLATION (if scripts don't work):
  1. Create virtual environment:
     python3 -m venv venv
  
  2. Activate it:
     Linux/Mac: source venv/bin/activate
     Windows: venv\Scripts\activate
  
  3. Install dependencies:
     pip install -r requirements.txt
  
  4. Run chatbot:
     python3 main.py

SYSTEM REQUIREMENTS:
  - Python 3.8 or higher
  - 100 MB disk space
  - Internet connection (first run only, for dependencies)

DATA FILES:
  - intents.csv: Training data for intent classification
  - qa.csv: Question-answer knowledge base
  - stock.json: Book inventory
  - locations.json: Store locations
  - orders.json: Order history
  - session.json: Session state
  - feedback.json: User feedback

CACHE FILES (auto-generated):
  *.pickle files are cached preprocessed data for faster startup.
  They auto-refresh every 24 hours.

For more information, see README.md
EOF

cd ..

# Create the ZIP archive
zip -r blacksmiths-bookstore-chatbot-portable.zip dist_portable/

echo ""
echo "==================================="
echo "Portable distribution created!"
echo "==================================="
echo ""
echo "Distribution package: blacksmiths-bookstore-chatbot-portable.zip"
echo ""
echo "To use:"
echo "  1. Unzip the package on any system"
echo "  2. Run ./run_chatbot.sh (Linux/Mac) or run_chatbot.bat (Windows)"
echo ""
