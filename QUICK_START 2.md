# Reflex-Strategy Quick Start Guide

Get the Reflex-Strategy System running in 5 minutes.

---

## Prerequisites

- Python 3.8 or higher
- (Optional) A YOLOv11 model for object detection
- Your game open and visible on screen

---

## Option 1: Use the Hosted VPS Brain

The VPS Brain is already running at `http://167.86.105.39:8001`

### Step 1: Install Dependencies

```bash
cd reflex-strategy-system
pip install -r requirements.txt
```

### Step 2: Create Configuration

```bash
cp .env.example .env
```

### Step 3: Configure for Your Game (minimal)

Edit `.env` and set:

```bash
# Point to the hosted VPS Brain
VPS_URL=ws://167.86.105.39:8001/ws

# Your game's potion key
KEY_POTION=space

# Your game's movement keys (if different from WASD)
KEY_MOVE_UP=w
KEY_MOVE_DOWN=s
KEY_MOVE_LEFT=a
KEY_MOVE_RIGHT=d

# IMPORTANT: Disable actuator for testing first
ENABLE_ACTUATOR=false
```

### Step 4: Run

```bash
python -m local.coordinator
```

You should see:
```
[COORDINATOR] Initializing Local PC (Spinal Cord)...
[COORDINATOR] Screen capture initialized
[COORDINATOR] Vision system initialized
[COORDINATOR] Instinct layer initialized
[COORDINATOR] Actuator initialized
[COORDINATOR] VPS client initialized
[COORDINATOR] All systems initialized
[COORDINATOR] Starting main loop...
```

### Step 5: Enable Automation

Once you see it running and detecting objects, edit `.env`:
```bash
ENABLE_ACTUATOR=true
```

Then restart the coordinator.

---

## Option 2: Self-Hosted Quick Setup

### VPS (Server) Setup

```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt update
sudo apt install postgresql postgresql-contrib postgis

# Create database
sudo -u postgres psql -c "CREATE DATABASE reflex_strategy;"
sudo -u postgres psql -d reflex_strategy -c "CREATE EXTENSION postgis;"

# Install dependencies
pip install -r requirements.txt

# Start VPS
DB_PASSWORD=your_password python -m vps.server
```

### Local PC Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create config
cat > .env << EOF
VPS_URL=ws://YOUR_VPS_IP:8000/ws
ENABLE_ACTUATOR=false
EOF

# Run
python -m local.coordinator
```

---

## Preset Configurations

Choose a preset and copy to your `.env` file.

### Preset 1: Passive Observation (Testing)

```bash
ENABLE_CAPTURE=true
ENABLE_VISION=true
ENABLE_INSTINCT=true
ENABLE_ACTUATOR=false
ENABLE_VPS_CONNECTION=true

DEBUG_MODE=true
LOG_DETECTIONS=true
```

Use this to verify detection works without affecting your game.

---

### Preset 2: Safe Farming

```bash
INSTINCT_CRITICAL_HP=40.0
INSTINCT_LOW_HP=60.0
INSTINCT_COMBAT_DISTANCE=120.0

ENABLE_ACTUATOR=true
DEBUG_MODE=false
LOG_DETECTIONS=false
```

Conservative settings for safer farming with earlier flight response.

---

### Preset 3: Aggressive Combat

```bash
INSTINCT_CRITICAL_HP=20.0
INSTINCT_LOW_HP=40.0
INSTINCT_COMBAT_DISTANCE=180.0

KEY_ATTACK=1
KEY_SKILL_1=q
KEY_SKILL_2=e

ENABLE_ACTUATOR=true
```

More aggressive combat with delayed retreat.

---

## Copy-Paste All Commands

### Windows (PowerShell)

```powershell
# Navigate to project
cd reflex-strategy-system

# Install dependencies
pip install -r requirements.txt

# Create config
Copy-Item .env.example .env

# Run
python -m local.coordinator
```

### Linux/Mac

```bash
cd reflex-strategy-system
pip install -r requirements.txt
cp .env.example .env
python -m local.coordinator
```

---

## Verify It's Working

### Check VPS Connection

```bash
curl http://167.86.105.39:8001/health
```

Expected output:
```json
{"status": "degraded" or "healthy", "connections": 0, ...}
```

### Check Database Has Data

```bash
curl "http://167.86.105.39:8001/db/nearest?x=0&y=0&z=0&object_type=market"
```

Should return market location data.

### Expected Local PC Output

```
[VISION] Found 2 detections
[INSTINCT] Security tier: SECURE
[COORDINATOR] Processing frame 1...
[COORDINATOR] FPS: 60.0
```

---

## Next Steps

1. **Read the full [USER_GUIDE.md](USER_GUIDE.md)** - For detailed configuration
2. **Train a YOLO model** - For your specific game's objects
3. **Adjust thresholds** - Fine-tune for your game's pace
4. **Read [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** - For advanced customization

---

## Support

For issues or questions:
- Check [USER_GUIDE.md](USER_GUIDE.md) - Troubleshooting section
- Enable `DEBUG_MODE=true` and `LOG_DETECTIONS=true` in `.env`
- Check logs: View terminal output or check git logs for VPS
