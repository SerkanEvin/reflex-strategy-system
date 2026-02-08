# Reflex-Strategy Developer Guide

Technical reference for extending and customizing the Reflex-Strategy System.

---

## Table of Contents

1. [Architecture](#architecture)
2. [REST API Reference](#rest-api-reference)
3. [WebSocket Protocol](#websocket-protocol)
4. [Data Models](#data-models)
5. [Extension Points](#extension-points)
6. [YOLO Integration](#YOLO-integration)
7. [LLM Integration](#llm-integration)

---

## Architecture

### Module Overview

```
/root/reflex-strategy-system/
├── shared/                 # Shared data models and constants
│   ├── models.py          # Dataclasses for state, commands, etc.
│   └── constants.py       # Enums, thresholds, key bindings
│
├── local/                 # Local PC (Spinal Cord)
│   ├── capture.py         # Screen capture (mss)
│   ├── vision.py          # YOLO inference
│   ├── instinct.py        # Security assessment
│   ├── actuator.py        # Human-like keyboard/mouse
│   ├── client.py          # WebSocket client
│   ├── coordinator.py     # Main orchestration loop
│   └── config.py          # Configuration classes
│
└── vps/                   # VPS Brain
    ├── server.py          # FastAPI server + REST endpoints
    ├── database.py        # PostgreSQL + PostGIS
    └── strategist.py      # AI-driven strategy generator
```

### Message Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│ Processing Frame (60 FPS)                                              │
│                                                                          │
│ 1. Capture Frame (capture.py)                                           │
│    └─> Returns: np.ndarray (BGR image, 1920x1080 or configured)      │
│                                                                          │
│ 2. YOLO Detection (vision.py)                                          │
│    └─> Returns: List[Detection] with bbox, label, confidence         │
│                                                                          │
│ 3. UI Detection (vision.py)                                            │
│    └─> Returns: PlayerState with HP, MP from screen                   │
│                                                                          │
│ 4. Security Assessment (instinct.py)                                    │
│    └─> Returns: SecurityTier + ReflexCommand                             │
│                                                                          │
│ 5. LocalState Build (coordinator.py)                                    │
│    └─> Combines: PlayerState + Detections + SecurityTier              │
│                                                                          │
│ 6. Decision Layer (coordinator.py)                                      │
│    ├─> If DANGER:    Execute reflex only                               │
│    ├─> If SECURE:    Check VPS strategy + merge decisions             │
│    └─> Otherwise:    Reflex can preempt strategy                      │
│                                                                          │
│ 7. Actuation (actuator.py)                                              │
│    └─> Execute with HumanMouse + KeyboardController                    │
│                                                                          │
│ 8. State Update (throttled, every 0.5s)                                │
│    └─> Send via WebSocket to VPS (client.py)                           │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ VPS Processing                                                          │
│                                                                          │
│ 1. Receive State (server.py)                                            │
│    └─> Parse LocalState                                                │
│                                                                          │
│ 2. Log to Database (database.py)                                       │
│    └─> Store in player_positions table                                 │
│                                                                          │
│ 3. Gather Context (strategist.py)                                      │
│    └─> spatial_memory: known locations                                 │
│        recent_positions: movement history                               │
│        current_strategy: existing plan                                  │
│                                                                          │
│ 4. Critical Rules Check (strategist.py)                                │
│    ├─> CRITICAL_HEALTH: RETREAT immediately                           │
│    ├─> FULL_INVENTORY: GO_TO_MARKET                                   │
│    ├─> HIGH_VALUE_TARGET: HUNT                                         │
│    └─> LOW_ON_POTIONS: FIND_VENDOR                                    │
│                                                                          │
│ 5. LLM Reasoning (strategist.py)                                       │
│    └─> If no critical rule: Run LLM or rule-based fallback             │
│                                                                          │
│ 6. Generate StrategyPolicy                                              │
│    ├─> action: "HUNT", "GO_TO_MARKET", etc.                           │
│    ├─> reasoning: Explanation                                          │
│    ├─> priority: 1-10                                                  │
│    └─> target: Optional location                                       │
│                                                                          │
│ 7. Log to Database                                                      │
│    └─> Store in strategy_logs table                                    │
│                                                                          │
│ 8. Send STRATEGY_UPDATE via WebSocket                                  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## REST API Reference

### Endpoints

#### GET `/`
Server information

**Response:**
```json
{
  "name": "Reflex-Strategy VPS Brain",
  "version": "1.0.0",
  "status": "running"
}
```

---

#### GET `/health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy" | "degraded",
  "timestamp": 1770491366.645257,
  "connections": 1,
  "database_connected": true
}
```

---

#### GET `/stats`
Server statistics

**Response:**
```json
{
  "connections": 1,
  "database": {
    "spatial_entries": 6,
    "total_player_positions": 1234,
    "active_sessions": 1,
    "entries_by_type": {
      "market": 1,
      "vendor": 2,
      "spawn": 1,
      "healer": 1,
      "safe_zone": 1
    }
  },
  "current_strategy": null | {
    "action": "PATROL",
    "reasoning": "...",
    "priority": 5,
    ...
  }
}
```

---

#### POST `/db/entry`
Create or update a spatial memory entry

**Request Body:**
```json
{
  "object_type": "market",
  "label": "central_market",
  "position": {"x": 1000.0, "y": 500.0, "z": 0.0},
  "last_seen": 1770491366.0,
  "confidence": 1.0,
  "metadata": {
    "type": "market",
    "vendors": ["weapons", "armor", "potions"]
  }
}
```

**Response:**
```json
{
  "entry_id": 7,
  "status": "created" | "updated"
}
```

---

#### GET `/db/entry/{object_type}/{label}`
Retrieve specific spatial entry

**Parameters:**
- `object_type` (path): Type of object
- `label` (path): Object identifier

**Response:**
```json
{
  "object_type": "market",
  "label": "central_market",
  "position": {"x": 1000.0, "y": 500.0, "z": 0.0},
  "last_seen": 1770491366.0,
  "confidence": 1.0,
  "metadata": {...}
}
```

**Error (404):** Entry not found

---

#### GET `/db/nearest`
Find nearest entries to a position

**Query Parameters:**
- `x` (float): X coordinate
- `y` (float): Y coordinate
- `z` (float): Z coordinate
- `object_type` (str): Type of object to search
- `limit` (int, default: 5): Max results
- `max_distance` (float, optional): Maximum distance

**Example:**
```
GET /db/nearest?x=1000&y=500&z=0&object_type=vendor&limit=2&max_distance=500
```

**Response:**
```json
[
  {
    "entry": {
      "object_type": "vendor",
      "label": "potion_vendor",
      "position": {"x": 1050.0, "y": 520.0, "z": 0.0},
      "last_seen": 1770491366.0,
      "confidence": 0.9,
      "metadata": {}
    },
    "distance": 52.0
  },
  {
    "entry": {...},
    "distance": 180.5
  }
]
```

---

#### DELETE `/db/entry/{object_type}/{label}`
Delete a spatial entry

**Parameters:**
- `object_type` (path): Type of object
- `label` (path): Object identifier

**Response:**
```json
{
  "deleted": 1,
  "status": "success"
}
```

---

#### POST `/strategy`
Request strategy for current state

**Request Body:**
```json
{
  "state": {
    "timestamp": 1770491366.0,
    "player": {
      "health_percent": 85.0,
      "max_health": 1000.0,
      "health": 850.0,
      "mana_percent": 100.0,
      "level": 15,
      "is_in_combat": false,
      "is_dead": false,
      "inventory_full_percent": 75.0,
      "position": {"x": 1000.0, "y": 500.0, "z": 0.0}
    },
    "detections": [
      {
        "bbox": {"x1": 100, "y1": 200, "x2": 200, "y2": 300, ...},
        "detection_type": "enemy",
        "label": "goblin",
        "confidence": 0.85,
        "distance": 50.0
      }
    ],
    "security_tier": 2,
    "active_targets": ["goblin"],
    "frame_id": 1234
  },
  "force_update": false
}
```

**Response (New Strategy):**
```json
{
  "policy_id": null,
  "priority": 4,
  "action": "PATROL",
  "target": {
    "type": "resource",
    "label": "iron_ore",
    "position": {"x": 1200.0, "y": 550.0, "z": 0.0}
  },
  "reasoning": "Safe area with resources nearby - patrolling",
  "estimated_duration": null,
  "conditions": null,
  "created_at": 1770491366.645257
}
```

**Response (No Change):**
```json
{
  "status": "no_change"
}
```

---

#### GET `/strategy/current`
Get current strategy policy

**Response (Has Policy):**
```json
{
  "policy_id": null,
  "priority": 4,
  "action": "PATROL",
  ...
}
```

**Response (No Policy):**
```json
{
  "status": "no_policy"
}
```

---

#### DELETE `/strategy/current`
Clear current strategy policy

**Response:**
```json
{
  "status": "cleared"
}
```

---

## WebSocket Protocol

### Connection

**Endpoint:** `ws://host:port/ws`

**Connect Message (Client → Server):**
```json
{
  "type": "CONNECT",
  "payload": {
    "client_type": "local_spinal_cord",
    "version": "1.0.0",
    "timestamp": 1770491366.0
  },
  "timestamp": 1770491366.0
}
```

**Connect Confirmation (Server → Client):**
```json
{
  "type": "CONNECT",
  "payload": {
    "status": "connected",
    "server_time": 1770491366.645257
  },
  "timestamp": 1770491366.0
}
```

---

### Message Types

#### STATE_UPDATE (Client → Server)
Send local state to VPS

```json
{
  "type": "STATE_UPDATE",
  "payload": {
    "timestamp": 1770491366.0,
    "player": {...},
    "detections": [...],
    "security_tier": 2,
    "active_targets": [...],
    "frame_id": 1234
  },
  "timestamp": 1770491366.0
}
```

#### STRATEGY_UPDATE (Server → Client)
New strategy from VPS

```json
{
  "type": "STRATEGY_UPDATE",
  "payload": {
    "policy_id": null,
    "priority": 4,
    "action": "GO_TO_MARKET",
    "target": {...},
    "reasoning": "...",
    "created_at": 1770491366.645257
  },
  "timestamp": 1770491366.0
}
```

#### POLICY_CONFIRMATION (Server → Client)
Acknowledge strategy was processed

```json
{
  "type": "POLICY_CONFIRMATION",
  "payload": {
    "policy_id": null,
    "state_received": true,
    "timestamp": 1770491366.0
  },
  "timestamp": 1770491366.0
}
```

#### HEARTBEAT (Client → Server)
Keep-alive message

```json
{
  "type": "HEARTBEAT",
  "payload": {
    "timestamp": 1770491366.0,
    "status": "alive"
  },
  "timestamp": 1770491366.0
}
```

#### PING / PONG (Bidirectional)
Connection health check

**PING:**
```json
{
  "type": "PING",
  "payload": {"timestamp": 1770491366.0}
}
```

**PONG:**
```json
{
  "type": "PONG",
  "payload": {"timestamp": 1770491366.0}
}
```

#### ERROR_REPORT (Client → Server)
Report errors to VPS

```json
{
  "type": "ERROR_REPORT",
  "payload": {
    "error": "Screen capture failed",
    "context": {"monitor_index": 1},
    "timestamp": 1770491366.0
  },
  "timestamp": 1770491366.0
}
```

#### DATABASE_QUERY (Server → Client)
Request data from client

```json
{
  "type": "DATABASE_QUERY",
  "payload": {
    "query_type": "get_ui_coordinates",
    "params": {...}
  },
  "timestamp": 1770491366.0
}
```

#### DATABASE_RESPONSE (Client → Server)

```json
{
  "type": "DATABASE_RESPONSE",
  "payload": {
    "query_id": "abc123",
    "data": {...},
    "timestamp": 1770491366.0
  },
  "timestamp": 1770491366.0
}
```

---

## Data Models

### SecurityTier (Enum)

```python
class SecurityTier(Enum):
    DANGER = 0
    ATTACK_MAY_OCCUR = 1
    SECURE = 2
```

### DetectionType (Enum)

```python
class DetectionType(Enum):
    ENEMY = "enemy"
    FRIENDLY = "friendly"
    PLAYER = "player"
    RESOURCE = "resource"
    ITEM = "item"
    UI_ELEMENT = "ui_element"
    MARKET = "market"
```

### BoundingBox

```python
@dataclass
class BoundingBox:
    x1: float          # Top-left X
    y1: float          # Top-left Y
    x2: float          # Bottom-right X
    y2: float          # Bottom-right Y
    confidence: float  # Detection confidence
    label: str         # Detection label

    @property
    center: Tuple[float, float]  # Center coordinates

    @property
    width: float
    @property
    height: float
```

### Detection

```python
@dataclass
class Detection:
    bbox: BoundingBox
    detection_type: DetectionType  # ENEMY, FRIENDLY, etc.
    label: str                   # e.g., "goblin", "orc"
    confidence: float
    distance: Optional[float]      # Estimated 3D distance
    health_percent: Optional[float]  # For enemies/players
```

### PlayerState

```python
@dataclass
class PlayerState:
    health_percent: float
    max_health: float
    health: float
    mana_percent: Optional[float]
    level: Optional[int]
    position: Optional[Dict[str, float]]  # {x, y, z} in game coords
    is_in_combat: bool
    is_dead: bool
    inventory_full_percent: Optional[float]
```

### LocalState

```python
@dataclass
class LocalState:
    timestamp: float
    player: PlayerState
    detections: List[Detection]
    security_tier: SecurityTier
    active_targets: List[str]  # Enemy names engaged
    frame_id: int
```

### ReflexCommand

```python
@dataclass
class ReflexCommand:
    command: Literal["POT", "FLEE", "ATTACK", "DEFEND", "NONE"]
    priority: int          # 0=Immediate, 10=Low
    reasoning: str
    target_id: Optional[str]
    parameters: Optional[Dict[str, Any]]
```

### StrategyPolicy

```python
@dataclass
class StrategyPolicy:
    policy_id: Optional[int]
    priority: int
    action: str              # "HUNT", "GATHER", "GO_TO_MARKET", etc.
    target: Optional[Dict[str, Any]]
    reasoning: str
    created_at: float
    estimated_duration: Optional[float]
    conditions: Optional[List[str]]
```

---

## Extension Points

### 1. Adding New Security Rules

**Location:** `vps/strategist.py` - `_initialize_critical_rules()`

**Example:**
```python
# Add to existing critical rules dict
"LOW_MANA": {
    "condition": lambda state: (
        state.player.mana_percent and
        state.player.mana_percent < 20.0
    ),
    "action": "FIND_VENDOR",
    "reasoning": "Low mana - need to restock mana potions",
    "priority": 2
}

"OVERWEIGHT": {
    "condition": lambda state: (
        state.player.inventory_full_percent and
        state.player.inventory_full_percent >= 95.0
    ),
    "action": "GO_TO_MARKET",
    "reasoning": "Overweight - must sell items immediately",
    "priority": 1
}
```

---

### 2. Adding New Strategy Actions

**Location:** `vps/strategist.py` - `_rule_based_strategy()`

**Example:**
```python
# Add to existing strategy logic
elif strategy == "FIND_VENDOR":
    return Strategy(
        "Find vendor to restock potions",
        "Searching for potion vendor...",
        3,
        target=self._find_nearest_vendor(context),
        conditions=["inventory_has_space"]
    )

elif strategy == "PATROL_SPAWN":
    return Strategy(
        "Patrol around spawn area",
        "Patrolling starting area for enemies",
        6,
        target=self._find_spawn_point(context)
    )
```

Also add to available actions in LLM prompt.

---

### 3. Adding New Detection Types

**Location:** `local/vision.py` - DetectionType enum

**Example:**
```python
class DetectionType(Enum):
    ENEMY = "enemy"
    FRIENDLY = "friendly"
    PLAYER = "player"
    RESOURCE = "resource"
    ITEM = "item"
    UI_ELEMENT = "ui_element"
    MARKET = "market"

    # Add new types:
    QUEST_NPC = "quest_npc"
    PORTAL = "portal"
    CHEST = "chest"
```

Then update YOLO categories:
```python
# In VisionSystem.__init__()
self.categories = {
    0: DetectionType.ENEMY,
    1: DetectionType.FRIENDLY,
    2: DetectionType.PLAYER,
    3: DetectionType.RESOURCE,
    4: DetectionType.ITEM,
    5: DetectionType.UI_ELEMENT,
    6: DetectionType.MARKET,
    7: DetectionType.QUEST_NPC,   # New
    8: DetectionType.PORTAL,      # New
    9: DetectionType.CHEST,        # New
}
```

---

### 4. Customizing Human-like Movement

**Location:** `local/actuator.py` - HumanMouse class

**Parameters:**
```python
BezierCurve(mouse_speed_range=(100, 300), jitter=2.0, pause_chance=0.1)
```

Example - Slower, more deliberate movements:
```python
HumanMouse(
    speed_range=(50, 150),     # Slower
    jitter=0.5,                  # Less random
    pause_chance=0.2            # More pauses
)
```

Example - Faster, more erratic:
```python
HumanMouse(
    speed_range=(300, 600),     # Faster
    jitter=5.0,                  # More random
    pause_chance=0.05            # Fewer pauses
)
```

---

### 5. Adding Custom Reflex Commands

**Location:** `local/instinct.py` - `_handle_danger()` and `_handle_potential_threat()`

**Example:**
```python
def _handle_danger(self, player_state, detections):
    # Add new command type
    if player_state.mana_percent < 20.0:
        return ReflexCommand(
            command="USE_MANA_POTION",  # New command
            priority=ReflexPriorities.IMMEDIATE,
            reasoning=f"Low mana {player_state.mana_percent:.1f}%",
            parameters={"potion_type": "mana"}
        )

    # Also update Actuator to handle this command
```

---

### 6. Integrating Different LLM Providers

**Location:** `vps/strategist.py` - `_call_llm()`

**OpenAI Example:**
```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=llm_api_key)

response = await client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a game automation strategist..."},
        {"role": "user", "content": prompt}
    ]
)
result = response.choices[0].message.content
```

**Anthropic Example:**
```python
import anthropic

client = anthropic.AsyncAnthropic(api_key=llm_api_key)

response = await client.messages.create(
    model="claude-3-opus-20240229",
    messages=[{"role": "user", "content": prompt}]
)
result = response.content[0].text
```

**Ollama (Local) Example:**
```python
import aiohttp

async with aiohttp.ClientSession() as session:
    response = await session.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama2",
            "prompt": prompt,
            "stream": False
        }
    )
result = (await response.json())["response"]
```

---

## YOLO Integration

### Model Requirements

**Format:** PyTorch `.pt` file trained with YOLOv11 architecture

**Classes:** Must include detections for:
- Class 0: ENEMY
- Class 1-6: Additional types (FRIENDLY, PLAYER, RESOURCE, ITEM, UI_ELEMENT, MARKET)

**Output Resolution:** Input images resized to model's native resolution (typically 640x640)

### Loading a Model

```python
from local.vision import VisionSystem

vision = VisionSystem(
    model_path="path/to/your_model.pt",
    confidence_threshold=0.5,
    device="cuda"  # or "cpu"
)

# Or load after initialization
vision.load_model("new_model.pt")
```

### Training Your Model

**1. Collect Data:**
```python
import cv2

# Capture 100 screenshots
for i in range(100):
    frame = capture.get_frame()
    cv2.imwrite(f"dataset/image_{i}.jpg", frame)
    time.sleep(1)
```

**2. Label Data:** Use Roboflow (https://roboflow.com) or LabelImg

**3. Export:** Export in YOLO format (`.txt` annotations with class_id x_center y_center width height)

**4. Train:**
```python
from ultralytics import YOLO

model = YOLO("yolov11n.pt")  # Start with pre-trained

model.train(
    data="your_data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="cuda"  # or "cpu"
)

# Export for use
model.export(format="torchscript")
```

### Using the Model

```python
# Run inference
detections = vision.detect(frame)

# Filter by type
enemies = [d for d in detections if d.detection_type == DetectionType.ENEMY]
resources = [d for d in detections if d.detection_type == DetectionType.RESOURCE]

# Visualize
vis_frame = vision.visualize_detections(frame, detections)
```

---

## LLM Integration

### Strategist LLM Role

The LLM is prompted to be a strategic AI for game automation:

```python
role = "You are a strategic AI brain for a game automation system."
```

### Available Actions for LLM

- `HUNT` - Hunt for enemies/targets
- `GATHER` - Collect resources
- `GO_TO_MARKET` - Travel to market/vendor
- `RETREAT` - Fall back to safe location
- `PATROL` - Patrol known area
- `EXPLORE` - Explore new areas
- `FIND_VENDOR` - Restock supplies

### Prompt Structure

```python
prompt = f"""
Current Situation:
- Security Level: {state.security_tier.name}
- Player Health: {state.player.health_percent}%
- In Combat: {state.player.is_in_combat}
- Inventory Full: {context.inventory_state.get('full_percent', 0)}%

Known Locations: {known_locations}

Current Policy: {context.current_strategy.action if context.current_strategy else "None"}

Determine the best high-level action. Response must be JSON format with action, reasoning, and priority.
"""
```

### Customizing LLM Prompts

**Location:** `vps/strategist.py` - `_build_llm_prompt()`

Add custom instructions:
```python
prompt += f"""
Additional Context:
- Game Type: 3D MMORPG
- Play Style: {self.play_style}  # Aggressive, Defensive, Balanced
- Time of Day: {time_of_day}

Available Actions: {self.available_actions}
```

---

## Database Schema

### spatial_memory

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| object_type | VARCHAR(50) | Type of object (market, vendor, etc.) |
| label | VARCHAR(100) | Object identifier |
| position | GEOMETRY(POINTZ, 4326) | 3D spatial coordinates (PostGIS) |
| last_seen | TIMESTAMP | When this location was last confirmed |
| confidence | FLOAT | Detection confidence (0-1) |
| metadata | JSONB | Additional properties |
| created_at | TIMESTAMP | Entry creation time |
| updated_at | TIMESTAMP | Last update time |

### player_positions

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| session_id | UUID | Unique session identifier |
| position | GEOMETRY(POINTZ, 4326) | Player's 3D position |
| health_percent | FLOAT | Current health |
| is_in_combat | BOOLEAN | Combat state |
| detected_objects | JSONB | Objects detected at this position |
| timestamp | TIMESTAMP | Recording time |

### strategy_logs

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL | Primary key |
| policy_id | INTEGER | Reference to policy |
| action | VARCHAR(100) | Action taken |
| reasoning | TEXT | Why this action was chosen |
| priority | INTEGER | Action priority |
| context | JSONB | Full game state snapshot |
| created_at | TIMESTAMP | When strategy was generated |

---

## Additional Resources

- [QUICK_START.md](QUICK_START.md) - 5-minute setup
- [USER_GUIDE.md](USER_GUIDE.md) - End-user documentation
- [README.md](README.md) - Project overview

---

## Contributing

When adding features:

1. Update this documentation
2. Add type hints
3. Include docstrings
4. Test with both enabled/disabled modules
5. Update .env.example if adding new config options

---

## License

MIT License
