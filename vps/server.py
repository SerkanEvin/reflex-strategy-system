"""
FastAPI WebSocket Server (VPS Brain)
Main server that handles connections from Local PC and coordinates
between database and strategist modules.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Set, Optional
import json
import uvicorn
from datetime import datetime
import asyncio

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from shared.models import (
    LocalState, WebSocketMessage, MessageType, StrategyPolicy, SpatialMemoryEntry
)
from vps.database import SpatialDatabase
from vps.strategist import Strategist


# Pydantic models for REST API endpoints
class DatabaseEntryRequest(BaseModel):
    object_type: str
    label: str
    position: Dict[str, float]
    last_seen: float
    confidence: float = 1.0
    metadata: Optional[Dict] = None


class DatabaseEntryResponse(BaseModel):
    entry_id: int
    status: str


class StrategyRequest(BaseModel):
    state: Dict
    force_update: bool = False


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    connections: int
    database_connected: bool


class VPSBrainServer:
    """
    Main FastAPI server for the VPS Brain.
    Handles WebSocket connections and coordinates subsystems.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        db_host: str = "localhost",
        db_port: int = 5432,
        db_name: str = "reflex_strategy",
        db_user: str = "postgres",
        db_password: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini"
    ):
        """
        Initialize VPS Brain server.

        Args:
            host: Server host address.
            port: Server port.
            db_host: Database host.
            db_port: Database port.
            db_name: Database name.
            db_user: Database user.
            db_password: Database password.
            llm_api_key: API key for LLM service.
            llm_model: LLM model to use.
        """
        self.host = host
        self.port = port

        # FastAPI app
        self.app = FastAPI(
            title="Reflex-Strategy VPS Brain",
            description="Cloud brain for game automation system",
            version="1.0.0"
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Subsystems
        self.database = SpatialDatabase(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password
        )

        self.strategist = Strategist(
            database=self.database,
            llm_api_key=llm_api_key,
            llm_model=llm_model
        )

        # WebSocket management
        self.active_connections: Set[WebSocket] = set()
        self.client_sessions: Dict[WebSocket, Dict] = {}

        # Register routes
        self._register_routes()

        # Task background processing
        self.background_tasks: Set[asyncio.Task] = set()

    def _register_routes(self):
        """Register all API routes."""
        # WebSocket endpoint
        self.app.add_api_websocket_route("/ws", self.websocket_endpoint)

        # REST API endpoints - use decorator-style registration
        self.app.get("/")(self.get_root)
        self.app.get("/health")(self.health_check)
        self.app.get("/stats")(self.get_statistics)

        # Database endpoints
        self.app.post("/db/entry")(self.create_database_entry)
        self.app.get("/db/entry/{object_type}/{label}")(self.get_database_entry)
        self.app.get("/db/nearest")(self.find_nearest_entries)
        self.app.delete("/db/entry/{object_type}/{label}")(self.delete_database_entry)

        # Strategy endpoints
        self.app.post("/strategy")(self.request_strategy)
        self.app.get("/strategy/current")(self.get_current_strategy)
        self.app.delete("/strategy/current")(self.clear_current_strategy)

    async def startup(self):
        """Startup tasks - initialize database and subsystems."""
        print(f"[VPS] Starting server on {self.host}:{self.port}")

        try:
            # Connect to database
            await self.database.connect()
            await self.database.initialize_schema()

            print("[VPS] Database connected and schema initialized")

        except Exception as e:
            print(f"[VPS] Database initialization failed: {e}")
            print("[VPS] Server may have limited functionality")

    def run(self):
        """Run the server."""
        # Run startup before starting server
        # Note: This is synchronous, we'll need async initialization

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )

    # WebSocket Handler
    async def websocket_endpoint(self, websocket: WebSocket):
        """
        Main WebSocket endpoint for client connections.

        Args:
            websocket: WebSocket connection.
        """
        await websocket.accept()
        self.active_connections.add(websocket)
        client_id = id(websocket)

        print(f"[VPS] Client {client_id} connected")

        # Initialize client session
        self.client_sessions[websocket] = {
            "client_id": client_id,
            "connected_at": datetime.now().timestamp(),
            "last_heartbeat": datetime.now().timestamp()
        }

        try:
            # Send acknowledgment
            ack_msg = WebSocketMessage(
                type=MessageType.CONNECT,
                payload={
                    "status": "connected",
                    "server_time": datetime.now().timestamp()
                }
            )
            await websocket.send_text(ack_msg.to_json())

            # Message loop
            while True:
                data = await websocket.receive_text()
                await self.handle_websocket_message(websocket, data)

        except WebSocketDisconnect:
            print(f"[VPS] Client {client_id} disconnected")
        except Exception as e:
            print(f"[VPS] Error with client {client_id}: {e}")
        finally:
            self.active_connections.discard(websocket)
            if websocket in self.client_sessions:
                del self.client_sessions[websocket]

    async def handle_websocket_message(self, websocket: WebSocket, data: str):
        """
        Handle incoming WebSocket message.

        Args:
            websocket: WebSocket connection.
            data: JSON message string.
        """
        try:
            # Parse message
            message = WebSocketMessage.from_json(data)

            # Update heartbeat
            if websocket in self.client_sessions:
                self.client_sessions[websocket]["last_heartbeat"] = datetime.now().timestamp()

            # Route by message type
            if message.type == MessageType.CONNECT:
                await self._handle_connect(websocket, message)

            elif message.type == MessageType.DISCONNECT:
                await self._handle_disconnect(websocket, message)

            elif message.type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(websocket, message)

            elif message.type == MessageType.PING:
                await self._handle_ping(websocket, message)

            elif message.type == MessageType.STATE_UPDATE:
                await self._handle_state_update(websocket, message)

            elif message.type == MessageType.ERROR_REPORT:
                await self._handle_error_report(websocket, message)

            elif message.type == MessageType.DATABASE_RESPONSE:
                await self._handle_database_response(websocket, message)

            else:
                print(f"[VPS] Unknown message type: {message.type}")

        except json.JSONDecodeError as e:
            print(f"[VPS] Invalid JSON message: {e}")
        except Exception as e:
            print(f"[VPS] Error handling message: {e}")

    async def _handle_connect(self, websocket: WebSocket, message: WebSocketMessage):
        """Handle client connect message."""
        payload = message.payload
        client_info = {
            "client_type": payload.get("client_type", "unknown"),
            "version": payload.get("version", "unknown"),
            "connected_at": datetime.now().timestamp()
        }

        if websocket in self.client_sessions:
            self.client_sessions[websocket].update(client_info)

        print(f"[VPS] Client {id(websocket)}: {client_info}")

    async def _handle_disconnect(self, websocket: WebSocket, message: WebSocketMessage):
        """Handle client disconnect message."""
        reason = message.payload.get("reason", "unknown")
        print(f"[VPS] Client {id(websocket)} disconnecting: {reason}")

    async def _handle_heartbeat(self, websocket: WebSocket, message: WebSocketMessage):
        """Handle client heartbeat."""
        if websocket in self.client_sessions:
            self.client_sessions[websocket]["last_heartbeat"] = datetime.now().timestamp()

    async def _handle_ping(self, websocket: WebSocket, message: WebSocketMessage):
        """Handle ping with pong."""
        pong_msg = WebSocketMessage(
            type=MessageType.PONG,
            payload={"timestamp": datetime.now().timestamp()}
        )
        await websocket.send_text(pong_msg.to_json())

    async def _handle_state_update(self, websocket: WebSocket, message: WebSocketMessage):
        """
        Handle state update from Local PC.

        Args:
            websocket: WebSocket connection.
            message: State update message.
        """
        try:
            # Parse local state
            state_dict = message.payload
            state = LocalState(**state_dict)

            # Feed to strategist for analysis
            policy = await self.strategist.analyze_and_decide(state)

            # Send strategy update if a new policy was generated
            if policy:
                strategy_msg = WebSocketMessage(
                    type=MessageType.STRATEGY_UPDATE,
                    payload=policy.to_dict()
                )
                await websocket.send_text(strategy_msg.to_json())

            # Send policy confirmation
            confirm_msg = WebSocketMessage(
                type=MessageType.POLICY_CONFIRMATION,
                payload={
                    "policy_id": policy.policy_id if policy else None,
                    "state_received": True,
                    "timestamp": datetime.now().timestamp()
                }
            )
            await websocket.send_text(confirm_msg.to_json())

        except Exception as e:
            print(f"[VPS] Error handling state update: {e}")

    async def _handle_error_report(self, websocket: WebSocket, message: WebSocketMessage):
        """Handle error report from client."""
        error = message.payload.get("error", "Unknown error")
        context = message.payload.get("context", {})
        print(f"[VPS] Error report from client {id(websocket)}: {error}")
        print(f"[VPS] Context: {context}")

    async def _handle_database_response(self, websocket: WebSocket, message: WebSocketMessage):
        """Handle database query response from client."""
        # Client-side queries can be processed here
        pass

    async def broadcast_message(self, message: WebSocketMessage):
        """
        Broadcast message to all connected clients.

        Args:
            message: Message to broadcast.
        """
        if self.active_connections:
            tasks = [
                ws.send_text(message.to_json())
                for ws in self.active_connections
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    # REST API Endpoints
    async def get_root(self):
        """Root endpoint."""
        return {
            "name": "Reflex-Strategy VPS Brain",
            "version": "1.0.0",
            "status": "running"
        }

    async def health_check(self) -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy" if self.database.pool else "degraded",
            timestamp=datetime.now().timestamp(),
            connections=len(self.active_connections),
            database_connected=self.database.pool is not None
        )

    async def get_statistics(self):
        """Get server statistics."""
        db_stats = await self.database.statistics() if self.database.pool else {}

        return {
            "connections": len(self.active_connections),
            "database": db_stats,
            "current_strategy": self.strategist.current_policy.to_dict() if self.strategist.current_policy else None
        }

    async def create_database_entry(self, request: DatabaseEntryRequest):
        """Create or update a spatial memory entry."""
        if not self.database.pool:
            raise HTTPException(status_code=503, detail="Database not connected")

        entry = SpatialMemoryEntry(
            entry_id=None,
            object_type=request.object_type,
            label=request.label,
            position=request.position,
            last_seen=request.last_seen,
            confidence=request.confidence,
            metadata=request.metadata
        )

        entry_id = await self.database.upsert_entry(entry)

        return DatabaseEntryResponse(
            entry_id=entry_id,
            status="created" if entry_id else "error"
        )

    async def get_database_entry(self, object_type: str, label: str):
        """Get a spatial memory entry."""
        entry = await self.database.get_entry_by_label(object_type, label)

        if entry is None:
            raise HTTPException(status_code=404, detail="Entry not found")

        return entry.to_dict()

    async def find_nearest_entries(
        self,
        x: float,
        y: float,
        z: float,
        object_type: str,
        limit: int = 5,
        max_distance: Optional[float] = None
    ):
        """Find nearest entries to a position."""
        # Database has fallback support, so we don't check pool
        position = {"x": x, "y": y, "z": z}
        results = await self.database.find_nearest(position, object_type, limit, max_distance)

        return [
            {
                "entry": entry.to_dict(),
                "distance": distance
            }
            for entry, distance in results
        ]

    async def delete_database_entry(self, object_type: str, label: str):
        """Delete a spatial memory entry."""
        if not self.database.pool:
            raise HTTPException(status_code=503, detail="Database not connected")

        async with self.database.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM spatial_memory WHERE object_type = $1 AND label = $2",
                object_type, label
            )

        deleted = int(result.split()[-1])
        return {"deleted": deleted, "status": "success" if deleted > 0 else "not_found"}

    async def request_strategy(self, request: StrategyRequest):
        """Request strategy for current state."""
        try:
            state = LocalState(**request.state)
            policy = await self.strategist.analyze_and_decide(state)

            if request.force_update and policy is None:
                # Force generate a new policy
                context = await self.strategist._gather_context(state)
                policy_dict = self.strategist._rule_based_strategy(state, context)
                policy = await self.strategist._finalize_policy(state, policy_dict)

            return policy.to_dict() if policy else {"status": "no_change"}

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def get_current_strategy(self):
        """Get current strategy policy."""
        if self.strategist.current_policy:
            return self.strategist.current_policy.to_dict()
        return {"status": "no_policy"}

    async def clear_current_strategy(self):
        """Clear current strategy policy."""
        self.strategist.reset()
        return {"status": "cleared"}


def main():
    """Main entry point."""
    import os

    # Configuration from environment variables
    config = {
        "host": os.getenv("VPS_HOST", "0.0.0.0"),
        "port": int(os.getenv("VPS_PORT", "8000")),
        "db_host": os.getenv("DB_HOST", "localhost"),
        "db_port": int(os.getenv("DB_PORT", "5432")),
        "db_name": os.getenv("DB_NAME", "reflex_strategy"),
        "db_user": os.getenv("DB_USER", "postgres"),
        "db_password": os.getenv("DB_PASSWORD"),
        "llm_api_key": os.getenv("LLM_API_KEY"),
        "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini")
    }

    # Create and run server
    server = VPSBrainServer(**config)

    # Run startup (will skip if database not available)
    try:
        import asyncio
        asyncio.run(server.startup())
    except:
        print("[VPS] Startup failed, running with limited functionality")

    # Run server (blocking)
    server.run()


if __name__ == "__main__":
    main()


# Module-level app for easy uvicorn import
# Import and use the main function instead
def get_app():
    """Get FastAPI app instance."""
    return VPSBrainServer().app

app = get_app()
