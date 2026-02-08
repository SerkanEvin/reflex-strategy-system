# Reflex-Strategy System

A hybrid architecture for game automation that combines immediate local reflexes with cloud-based strategic planning.

## What Is This Tool?

The Reflex-Strategy System is a sophisticated game automation framework that mimics human-like behavior while maintaining split-second reaction times. It's designed for 3D MMORPGs and similar games.

### What It Does

**Automates:**
- **Combat**: Detects enemies, assesses threats, fights or flees appropriately
- **Resource Gathering**: Finds and collects game resources efficiently
- **Inventory Management**: Automatically sells items and restocks supplies
- **Movement**: Uses natural, human-like mouse and keyboard patterns
- **Strategy**: Makes intelligent long-term decisions using AI/LLM

**How It Works:**
- **Local PC (Spinal Cord)**: Captures your screen at 60 FPS, detects objects in real-time, and executes reflex actions with zero latency
- **VPS Brain (Cloud)**: Analyzes game state, remembers locations, and generates strategic plans
- **Hybrid Decision**: When danger appears, local reflexes take over immediately. When safe, cloud strategy guides behavior

### Why Hybrid Architecture?

| Component | Latency | Purpose |
|-----------|---------|---------|
| **Local PC** | 0ms | Immediate reactions (dodge, heal, attack) |
| **VPS Brain** | ~50-200ms | Long-term planning (route to market, optimize farming pattern) |

This separation ensures you never miss a critical action while benefiting from intelligent planning.

---

## Quick Start (5 Minutes)

### Option 1: Use the VPS Brain (Already Running)

The VPS Brain is already running at `http://167.86.105.39:8001`

**On your Local PC:**

```bash
# Clone or navigate to the project
cd reflex-strategy-system

# Install dependencies
pip install -r requirements.txt

# Configure for your game
cp .env.example .env
# Edit .env with your game settings (see USER_GUIDE.md)

# Run the Local PC (Spinal Cord)
python -m local.coordinator
```

### Option 2: Full Self-Hosted Setup

**On your VPS/Server:**

```bash
# Setup the database (PostgreSQL)
python setup_local.py --host localhost --password YOUR_PASSWORD

# Start the VPS Brain
python -m vps.server
```

**Then on your Local PC** (see above).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         VPS (The Brain)                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Database  │    │  Strategist │    │   WebSocket │         │
│  │ (PostgreSQL)│───▶│    (LLM)    │◀──▶│   Server    │         │
│  │   + PostGIS │    │   Policies  │    │  (FastAPI)  │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
└─────────────────────────────┬───────────────────────────────────┘
                              │ WebSocket
                              │ (Bi-directional)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Local PC (Spinal Cord)                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Capture    │───▶│   Vision    │───▶│  Instinct   │         │
│  │   (mss)     │    │   (YOLO)    │    │  Security   │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│                                              │                  │
│  ┌─────────────┐    ┌─────────────┐         ▼                  │
│  │   Actuator  │◀───│ Coordinator │    ┌──────────┐           │
│  │  (Human-like)│   │  (Main Loop)│    │  Client  │           │
│  └─────────────┘    └─────────────┘    └──────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### Local PC (Spinal Cord)

| Module | Description |
|--------|-------------|
| **Capture** (`capture.py`) | Screen capture at 60 FPS using mss library. Zero-latency frame streaming. |
| **Vision** (`vision.py`) | YOLOv11 object detection. Recognizes enemies, allies, resources, items, and UI elements. |
| **Instinct** (`instinct.py`) | Security assessment layer. Classifies environment into DANGER, ATTACK_MAY_OCCUR, or SECURE tiers. |
| **Actuator** (`actuator.py`) | Human-like keyboard/mouse input using Bezier curves and randomized delays. Evades detection. |
| **Client** (`client.py`) | WebSocket client. Streams state to VPS and receives strategy commands. |
| **Coordinator** (`coordinator.py`) | Main orchestration loop. Merges reflex and strategy for execution. |

### VPS (Brain)

| Module | Description |
|--------|-------------|
| **Database** (`database.py`) | PostgreSQL + PostGIS. Stores game locations (markets, spawns), tracks player movement, logs decisions. |
| **Strategist** (`strategist.py`) | AI-powered strategic planner. Uses LLM for complex decisions, rule-based fallbacks for emergencies. |
| **Server** (`server.py`) | FastAPI WebSocket server. Handles connections, coordinates subsystems, provides REST API. |

---

## Installation

### For Local PC

```bash
# Clone the repository
git clone http://167.86.105.39/root/game-automation.git
cd game-automation

# Install Python dependencies
pip install -r requirements.txt

# For YOLO model support (optional but recommended)
pip install ultralytics torch torchvision

# Create configuration file
cp .env.example .env
# Edit .env with your settings (see Configuration)
```

### For VPS

```bash
# Install PostgreSQL with PostGIS
sudo apt update
sudo apt install postgresql postgresql-contrib postgis

# Install Python dependencies
pip install -r requirements.txt

# Initialize database
python setup_local.py --host localhost --password YOUR_PASSWORD

# Or use SQL directly
psql -U postgres -f setup_database.sql
```

### Docker (Alternative)

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f reflex-strategy-vps
```

---

## Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

### Key Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `VISION_MODEL_PATH` | - | Path to your YOLO model weights |
| `INSTINCT_CRITICAL_HP` | 30.0 | Health % that triggers DANGER tier (reflex only) |
| `INSTINCT_COMBAT_DISTANCE` | 150.0 | Distance (pixels) to consider "nearby enemy" |
| `KEY_POTION` | space | Keyboard key to use healing potions |
| `VPS_URL` | ws://167.86.105.39:8000/ws | WebSocket URL to VPS Brain |

### Key Bindings

| Action | Default Key |
|--------|-------------|
| Move Forward | w |
| Move Backward | s |
| Strafe Left | a |
| Strafe Right | d |
| Attack | 1 |
| Skill 1 | q |
| Skill 2 | e |
| Skill 3 | r |
| Use Potion | space |
| Interact | f |
| Sprint | shift |

---

## Running

### Start the VPS Server

```bash
# Set environment variables or use .env
export DB_PASSWORD=your_password
export LLM_API_KEY=your_key

# Run server
python -m vps.server

# Or use the startup script
./start_vps.sh
```

### Start the Local PC Coordinator

```bash
# Set environment variables or use .env
export VPS_URL=ws://your-vps:8000/ws

# Run coordinator
python -m local.coordinator

# Or use the startup script
./start_local.sh
```

### Using the Management Script (VPS)

```bash
# Quick commands
./manage.sh start       # Start VPS service
./manage.sh stop        # Stop VPS service
./manage.sh restart     # Restart VPS service
./manage.sh status      # Show detailed status
./manage.sh test        # Test all API endpoints
./manage.sh info        # Show system information

# View logs
./manage.sh logs
```

---

## Security Tiers

| Tier | Priority | Trigger Conditions | Behavior |
|------|----------|-------------------|----------|
| **DANGER** | 0 (Highest) | Health ≤30%, In combat, Enemies within 150px | Reflex ONLY - immediate action |
| **ATTACK_MAY_OCCUR** | 1 | Health ≤50%, Recent threats, Enemies 150-300px away | Reflex can preempt strategy |
| **SECURE** | 2 (Lowest) | No threats detected | Full strategy execution |

### Reflex Commands

The system can automatically execute these when needed:

| Command | When It Triggers | Action |
|---------|------------------|--------|
| **POT** | Health is critical (≤30%) or low (≤50%) | Press potion key to heal |
| **FLEE** | Danger detected, enemy too strong | Sprint and strafe away to escape |
| **ATTACK** | Weak enemy detected, in safe position | Engage with attack rotation |
| **DEFEND** | In combat but no target | Take defensive stance, block/dodge |

---

## What It Automates

### Combat
- Detects enemies using YOLO computer vision
- Assesses enemy strength and player health
- Decides to fight or flee automatically
- Uses natural attack rotations (skill combos)

### Resource Gathering
- Finds resource nodes in the environment
- Approach and interact efficiently
- Returns to safe area when inventory is full

### Inventory Management
- Automatically visits markets when inventory >90% full
- Routes to nearest vendor based on spatial memory
- Uses spatial memory to recall vendor locations

### Movement
- Human-like mouse movement using Bezier curves
- Randomized typing delays to look natural
- Evasion of server-side detection systems

---

## API Reference (Quick)

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Server information |
| GET | `/health` | Health check |
| GET | `/stats` | Server statistics |
| GET | `/db/nearest?x=&y=&z=&object_type=` | Find nearest game location |
| POST | `/strategy` | Request new strategy from VPS |

### WebSocket

- **Endpoint**: `ws://host:port/ws`
- **Messages**: State updates, strategy commands, heartbeats
- **Documentation**: See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)

---

## Documentation

| Document | Purpose |
|----------|---------|
| **[QUICK_START.md](QUICK_START.md)** | Get running in 5 minutes |
| **[USER_GUIDE.md](USER_GUIDE.md)** | Complete user manual, troubleshooting |
| **[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)** | API reference, extending the system |

---

## Troubleshooting

### Common Issues

**"YOLO model not found"** → Set `VISION_MODEL_PATH` in .env, or use fallback color detection

**"Database connection failed"** → Check VPS is running at configured URL

**"Capture screen is black"** → Check screen permissions, adjust `CAPTURE_MONITOR_INDEX`

**"Nothing happens when running"** → Verify `ENABLE_ACTUATOR=true` in .env

For full troubleshooting guide, see [USER_GUIDE.md](USER_GUIDE.md).

---

## Flow

1. **Perception**: Capture frame → YOLO detection → Player state estimation (HP/MP from UI)
2. **Instinct**: Assess security tier → Generate reflex command
3. **Decision**: If danger, execute reflex. Else, check VPS strategy
4. **Actuation**: Execute with human-like movement
5. **Feedback**: Send state to VPS → Close the loop

---

## License

MIT License

---

## Git Repository

**URL**: http://167.86.105.39/root/game-automation

Current branch: `main`
