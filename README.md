# Reflex-Strategy System

A hybrid architecture for game automation that combines immediate local reflexes with cloud-based strategic planning.

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

## Components

### Local PC (Spinal Cord)

1. **Capture (`capture.py`)** - Screen capture using mss library
   - Zero-latency frame capture
   - Background thread with buffer

2. **Vision (`vision.py`)** - YOLOv11 inference
   - Object detection (enemies, resources, UI)
   - Depth estimation for 3D games

3. **Instinct (`instinct.py`)** - Security assessment
   - Three-tier security: DANGER, ATTACK_MAY_OCCUR, SECURE
   - Immediate reflex triggers (POT, FLEE, ATTACK)

4. **Actuator (`actuator.py`)** - Human-like movement
   - Bezier curve mouse movement
   - Randomized delays for anti-detection

5. **Client (`client.py`)** - WebSocket client
   - Bi-directional communication with VPS

### VPS (Brain)

1. **Database (`database.py`)** - PostgreSQL + PostGIS
   - Spatial memory of game locations
   - Player position tracking
   - Strategy logging

2. **Strategist (`strategist.py`)** - LLM reasoning
   - High-level strategic planning
   - Rule-based fallbacks

3. **Server (`server.py`)** - FastAPI WebSocket server
   - Handles client connections
   - Coordinates subsystems

## Installation

### For Local PC

```bash
pip install -r requirements.txt
```

### For VPS

```bash
# Install PostgreSQL with PostGIS
sudo apt update
sudo apt install postgresql postgresql-contrib postgis

# Install Python dependencies
pip install -r requirements.txt

# Initialize database
psql -U postgres -f setup_database.sql
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key configuration options:
- Camera/game window settings
- YOLO model paths
- Security thresholds
- Key bindings
- VPS connection URL

## Running

### Start the VPS Server

```bash
# Set environment variables or use .env
export DB_PASSWORD=your_password
export LLM_API_KEY=your_key

# Run server
python -m vps.server
```

### Start the Local PC Coordinator

```bash
# Set environment variables or use .env
export VPS_URL=ws://your-vps:8000/ws

# Run coordinator
python -m local.coordinator
```

## Security Tiers

| Tier | Description | Behavior |
|------|-------------|----------|
| DANGER | Active combat or critical health | Reflex only - immediate action |
| ATTACK_MAY_OCCUR | Hostiles nearby | Reflex can preempt strategy |
| SECURE | No threats | Full strategy execution |

## Flow

1. **Perception**: Capture frame → YOLO detection → Player state estimation
2. **Instinct**: Assess security → Generate reflex command
3. **Decision**: If danger, execute reflex. Else, check VPS strategy
4. **Actuation**: Execute with human-like movement
5. **Feedback**: New state → Close the loop

## License

MIT License
