# Reflex-Strategy User Guide

Complete guide for using the Reflex-Strategy System.

---

## Table of Contents

1. [Understanding the System](#chapter-1-understanding-the-system)
2. [Installation](#chapter-2-installation)
3. [Configuration](#chapter-3-configuration)
4. [Running the System](#chapter-4-running)
5. [Troubleshooting](#chapter-5-troubleshooting)

---

## Chapter 1: Understanding the System

### 1.1 What Gets Automated vs. What You Control

| Automated | You Control |
|-----------|-------------|
| Enemy detection | Which YOLO model to use |
| Threat assessment | Security thresholds (HP, distance) |
| Reflex actions (heal, flee) | Key bindings |
| Strategic navigation | When to enable/disable actuation |
| Inventory management | Game window to capture |

### 1.2 The Hybrid Architecture Explained

```
Local PC (Your Machine)                           VPS Brain (Cloud)
┌─────────────────────────────┐                   ┌──────────────────┐
│ 1. Capture Screen            │                   │                  │
│    - 60 FPS capture           │───── State ───────▶│ 1. Store in     │
│    - Detect enemies           │   Update          │    Database      │
│                             │                   │                  │
│ 2. Assess Danger             │                   │ 2. Analyze with  │
│    - Check health            │                   │    AI/LLM         │
│    - Count nearby enemies    │◀─── Strategy ─────│                  │
│    - Generate reflex action   │   Response        │                  │
└─────────────────────────────┘                   └──────────────────┘
```

**If Danger Detected**: Local PC acts immediately (0ms delay)
**If Safe Local Area**: VPS Brain guides behavior (~50-200ms delay)

### 1.3 Message Flow

```
Your Frame (60/sec)
    ↓
YOLO Detection → Found: 2 enemies, 1 resource
    ↓
Instinct Layer → Security: SECURE
    ↓
Coordinator → Send to VPS → Get Strategy: PATROL
    ↓
Actuator → Move with human-like pattern
```

---

## Chapter 2: Installation

### 2.1 Local PC Setup

#### Windows

```powershell
# Install Python from python.org (Python 3.8+)

# Clone repository (or download zip)
git clone http://167.86.105.39/root/game-automation.git
cd game-automation

# Install dependencies
pip install -r requirements.txt
```

#### Linux

```bash
# Install Python 3
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Clone repository
git clone http://167.86.105.39/root/game-automation.git
cd game-automation

# Install dependencies
pip3 install -r requirements.txt
```

#### Mac

```bash
# Install Python 3 with Homebrew
brew install python3

# Clone repository
git clone http://167.86.105.39/root/game-automation.git
cd game-automation

# Install dependencies
pip3 install -r requirements.txt
```

### 2.2 VPS Brain Setup

#### Option A: Use Hosted VPS (Recommended)

The VPS Brain is already running at `http://167.86.105.39:8001`

Just configure your `.env` file:
```bash
VPS_URL=ws://167.86.105.39:8001/ws
```

#### Option B: Self-Hosted

**Ubuntu/Debian:**

```bash
# Install PostgreSQL with PostGIS
sudo apt update
sudo apt install postgresql postgresql-contrib postgis

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE reflex_strategy;
\c reflex_strategy
CREATE EXTENSION postgis;
\q
EOF

# Install Python dependencies
pip install -r requirements.txt

# Run setup script
python setup_local.py --host localhost --password YOUR_PASSWORD
```

**Or use SQL:**

```bash
psql -U postgres -f setup_database.sql
```

Then start the VPS server:
```bash
python -m vps.server
```

### 2.3 Option: Using Docker

```bash
# Build and run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f reflex-strategy-vps

# Stop
docker-compose down
```

---

## Chapter 3: Configuration

### 3.1 Creating Your Configuration File

Copy the example and edit it:
```bash
cp .env.example .env
nano .env  # or use your preferred editor
```

### 3.2 Screen Capture Settings

```bash
# Which monitor to capture (1 = primary display)
CAPTURE_MONITOR_INDEX=1

# Target FPS (higher = more responsive, more CPU)
CAPTURE_FPS=60

# Capture specific window instead of fullscreen
CAPTURE_WINDOW=false
CAPTURE_WINDOW_TITLE=YourGameWindowName

# Use grayscale for faster processing
CAPTURE_GRAYSCALE=false
```

**Finding your window title:**
- Windows: Use Task Manager or Alt-Tab
- Linux: Use `wmctrl -l`
- Mac: Use CMD+Tab or Activity Monitor

### 3.3 YOLO Vision Settings

```bash
# Path to your trained YOLO model
VISION_MODEL_PATH=path/to/your_model.pt

# Detection confidence (0.0 to 1.0)
# Higher = fewer but more confident detections
VISION_CONFIDENCE=0.5

# NMS IoU threshold
VISION_IOU_THRESHOLD=0.45

# Use CPU or GPU acceleration
VISION_DEVICE=cpu  # or "cuda" for NVIDIA GPU

# Estimate 3D distance from 2D screen coordinates
VISION_ESTIMATE_DEPTH=true

# Minimum detection size in pixels
VISION_MIN_SIZE=20
```

**Note**: If you don't have a YOLO model, the system uses fallback color-based detection (red HP bars, circular minimap).

### 3.4 Instinct (Security) Settings

```bash
# Health percentage that triggers DANGER (reflex only)
INSTINCT_CRITICAL_HP=30.0

# Health percentage that triggers ATTACK_MAY_OCCUR
INSTINCT_LOW_HP=50.0

# Distance in pixels to consider someone "nearby"
INSTINCT_COMBAT_DISTANCE=150.0

# Distance in pixels to consider area "safe"
INSTINCT_SAFE_DISTANCE=300.0

# Seconds between reflex actions
INSTINCT_REFLEX_COOLDOWN=1.0

# Seconds to remember threats after they leave screen
INSTINCT_THREAT_DECAY=10.0
```

**Adjust for your game:**
- Fast-paced combat: Lower critical HP (20%), higher reflex_cooldown (0.5s)
- Slow-paced: Higher critical HP (40%), lower reflex_cooldown (2s)

### 3.5 Key Bindings

```bash
# Movement
KEY_MOVE_UP=w
KEY_MOVE_DOWN=s
KEY_MOVE_LEFT=a
KEY_MOVE_RIGHT=d
KEY_SPRINT=shift

# Combat
KEY_ATTACK=1
KEY_SKILL_1=q
KEY_SKILL_2=e
KEY_SKILL_3=r

# Misc
KEY_POTION=space
KEY_INTERACT=f
```

**Mapping to your game keys:**
1. Find your game's key bindings settings
2. Map them to the corresponding action
3. Test with `ENABLE_ACTUATOR=true` and `DEBUG_MODE=true` to verify

### 3.6 VPS Connection Settings

```bash
# WebSocket URL to VPS Brain
VPS_URL=ws://167.86.105.39:8000/ws

# Reconnect after X seconds if connection lost
VPS_RECONNECT_INTERVAL=5.0

# Send heartbeat every X seconds
VPS_HEARTBEAT_INTERVAL=30.0

# Connection timeout
VPS_CONNECTION_TIMEOUT=10.0
```

### 3.7 Feature Flags

```bash
# Enable/disable individual components
ENABLE_CAPTURE=true
ENABLE_VISION=true
ENABLE_INSTINCT=true
ENABLE_ACTUATOR=true
ENABLE_VPS_CONNECTION=true
```

**For testing:**
- Set `ENABLE_ACTUATOR=false` to see what it would do without affecting the game
- Set `LOG_DETECTIONS=true` to see what objects are detected

### 3.8 Debug Settings

```bash
# Enable debug logging
DEBUG_MODE=false

# Log every detection made
LOG_DETECTIONS=false

# Show detection overlay on captured frames
VISUALIZE=false
```

---

## Chapter 4: Running the System

### 4.1 Starting the VPS Brain

```bash
# Set environment variables (or use .env)
export DB_PASSWORD=your_password
export LLM_API_KEY=your_key  # Optional, for AI strategies

# Start server
python -m vps.server

# Using the startup script
./start_vps.sh

# As a systemd service (persistent)
./manage.sh start
```

### 4.2 Starting the Local PC

```bash
# Set environment variables
export VPS_URL=ws://167.86.105.39:8000/ws

# Run coordinator
python -m local.coordinator

# Using the startup script
./start_local.sh
```

### 4.3 Understanding the Output

**Normal startup:**
```
[COORDINATOR] Initializing Local PC (Spinal Cord)...
[COORDINATOR] Screen capture initialized
[COORDINATOR] Vision system initialized
[COORDINATOR] Instinct layer initialized
[COORDINATOR] Actuator initialized
[COORDINATOR] VPS client initialized
[COORDINATOR] All systems initialized
[COORDINATOR] Starting main loop...
[CLIENT] Connected to VPS Brain
```

**During operation:**
```
[VISION] Found 3 detections
[INSTINCT] Security tier: ATTACK_MAY_OCCUR
[COORDINATOR] Generated reflex: POT with priority 1
[COORDINATOR] New strategy: PATROL
[COORDINATOR] FPS: 60.0
```

### 4.4 Stopping the System

**Local PC:**
```bash
# Press Ctrl+C in the terminal
# Or run:
pkill -f "python -m local.coordinator"
```

**VPS Brain:**
```bash
# Using management script
./manage.sh stop

# Or kill process
pkill -f "python -m vps.server"

# Using systemd
systemctl stop reflex-strategy-vps
```

---

## Chapter 5: Troubleshooting

### 5.1 Common Issues and Solutions

#### "YOLO model not found"

**Cause**: No model path specified in configuration.

**Solution**:
```bash
# Option A: Point to your model
VISION_MODEL_PATH=path/to/model.pt

# Option B: Use fallback mode (no model needed)
# Just leave VISION_MODEL_PATH empty
# System will use color-based detection for HP bars, minimap
```

---

#### "Database connection failed"

**Cause**: VPS not running or wrong URL.

**Solution**:
```bash
# Check VPS is running
curl http://167.86.105.39:8001/health

# Check your .env file has correct URL
VPS_URL=ws://167.86.105.39:8001/ws
```

---

#### "Capture screen is black"

**Cause**: Screen capture permissions or wrong monitor index.

**Solution**:
```bash
# Linux: Grant screen permissions
xhost +

# Try different monitor index
CAPTURE_MONITOR_INDEX=1  # Try 2, 3, etc.

# For Linux, ensure DISPLAY is set
export DISPLAY=:0
```

---

#### "Nothing happens when running"

**Cause**: Actuator disabled or game window not active.

**Solution**:
```bash
# Enable actuator
ENABLE_ACTUATOR=true

# Set DEBUG_MODE to verify detection
DEBUG_MODE=true
LOG_DETECTIONS=true

# Make sure your game window is active/focused
# The system can only send keystrokes to the active window
```

---

#### "Detected wrong objects"

**Cause**: Not using correct YOLO model for your game.

**Solution**:
1. Train a YOLO model on your game screenshots
2. Save as `.pt` file
3. Set `VISION_MODEL_PATH=path/to/model.pt`
4. Restart the coordinator

---

#### "System uses my mouse/keyboard uncontrollably"

**Cause**: Actuator enabled while testing.

**Solution**:
```bash
# Disable actuator
ENABLE_ACTUATOR=false

# Or use a separate test window that doesn't matter if moved
```

---

### 5.2 Getting Logs

#### Local PC Logs

The system prints directly to terminal when running:
```bash
python -m local.coordinator
```

#### VPS Brain Logs

```bash
# View systemd service logs
./manage.sh logs

# Or systemd
journalctl -u reflex-strategy-vps -f

# Application logs
tail -f /var/log/reflex-strategy-vps.log
tail -f /var/log/reflex-strategy-vps-error.log
```

#### Debug Mode

Enable in `.env`:
```bash
DEBUG_MODE=true
LOG_DETECTIONS=true
```

This will output:
- Every frame processed
- Every object detected
- Security tier for each frame
- Reflex and strategy decisions

### 5.3 Testing Without Affecting Game

Start with these settings:
```bash
ENABLE_ACTUATOR=false
DEBUG_MODE=true
LOG_DETECTIONS=true
```

Run the system and observe the output. Once you're satisfied with detections:
1. Toggle `ENABLE_ACTUATOR=true`
2. Toggle `DEBUG_MODE=false`
3. Restart

### 5.4 Performance Issues

**High CPU usage:**
```bash
# Lower FPS
CAPTURE_FPS=30

# Increase detection threshold
VISION_CONFIDENCE=0.6

# Use grayscale capture
CAPTURE_GRAYSCALE=true
```

**High memory usage:**
```bash
# Reduce buffer size
CAPTURE_BUFFER_SIZE=3
```

### 5.5 Advanced: Using YOLO Model

#### Training Your Model

1. **Collect Screenshots**: Capture 100+ screenshots of your game
2. **Label Images**: Use a tool like Roboflow or LabelImg
3. **Export**: Export in YOLO format
4. **Train**:
```bash
pip install ultralytics torch

# Your training code...
yolo train data=your_data.yaml model=yolov11n.pt epochs=100
```

#### Model Format

Must be a PyTorch `.pt` file trained with YOLOv11 architecture.

Detection classes must include:
- Class 0: ENEMY
- Class 1: FRIENDLY
- Class 2: PLAYER
- Class 3: RESOURCE
- Class 4: ITEM
- Class 5: UI_ELEMENT
- Class 6: MARKET (optional)

---

## 6. Getting Help

### Check Documentation

- **[README.md](README.md)** - Quick overview
- **[QUICK_START.md](QUICK_START.md)** - 5-minute setup
- **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** - Technical documentation

### Debug Checklist

- [ ] Dependencies installed: `pip list | grep -E "(mss|opencv|fastapi)"`
- [ ] Configuration file exists: `ls .env`
- [ ] VPS reachable: `curl http://167.86.105.39:8001/health`
- [ ] Screen permissions on Linux: `xhost +`
- [ ] Game window visible and active
- [ ] Debug mode enabled for testing

---

## 7. Configuration Examples

### Example 1: Conservative Farming

```bash
# .env for passive, safe resource gathering
INSTINCT_CRITICAL_HP=50.0
INSTINCT_LOW_HP=70.0
INSTINCT_COMBAT_DISTANCE=100.0
INSTINCT_REFLEX_COOLDOWN=2.0

ENABLE_ACTUATOR=true
DEBUG_MODE=false
LOG_DETECTIONS=false

# Slow, cautious movement
CAPTURE_FPS=30
```

### Example 2: Aggressive Combat

```bash
# .env for fast combat action
INSTINCT_CRITICAL_HP=20.0
INSTINCT_LOW_HP=40.0
INSTINCT_COMBAT_DISTANCE=200.0
INSTINCT_REFLEX_COOLDOWN=0.5

KEY_ATTACK=1
KEY_SKILL_1=q
KEY_SKILL_2=e
KEY_SKILL_3=r

ENABLE_ACTUATOR=true
DEBUG_MODE=false

# Fast detection and reaction
CAPTURE_FPS=60
VISION_CONFIDENCE=0.4
```

### Example 3: Testing/Observation Only

```bash
# .env for testing without game interaction
ENABLE_ACTUATOR=false
DEBUG_MODE=true
LOG_DETECTIONS=true
VISUALIZE=false

# High confidence to reduce false positives
VISION_CONFIDENCE=0.7
```
