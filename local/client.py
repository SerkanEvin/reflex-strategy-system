"""
WebSocket Client - Connects Local PC to VPS Brain
Handles communication and synchronization with the cloud brain.
"""
import asyncio
import json
import time
from typing import Optional, Callable, Dict, Any
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from shared.models import (
    LocalState, WebSocketMessage, MessageType, StrategyPolicy
)


class VPSClient:
    """
    WebSocket client for communication with the VPS Brain.
    Handles bi-directional message passing.
    """

    def __init__(
        self,
        vps_url: str = "ws://167.86.105.39:8000/ws",
        reconnect_interval: float = 5.0,
        heartbeat_interval: float = 30.0
    ):
        """
        Initialize VPS client.

        Args:
            vps_url: WebSocket URL of VPS server.
            reconnect_interval: Seconds between reconnection attempts.
            heartbeat_interval: Seconds between heartbeat messages.
        """
        self.vps_url = vps_url
        self.reconnect_interval = reconnect_interval
        self.heartbeat_interval = heartbeat_interval

        # Connection state
        self.connected = False
        self.websocket = None
        self._running = False
        self._listener_task = None
        self._heartbeat_task = None

        # Message handlers
        self.on_strategy_update: Optional[Callable[[StrategyPolicy], None]] = None
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None
        self.on_database_query: Optional[Callable[[Dict[str, Any]], None]] = None

        # Latest strategy cache
        self.latest_strategy: Optional[StrategyPolicy] = None

    async def connect(self):
        """Connect to VPS WebSocket server."""
        while self._running:
            try:
                print(f"[CLIENT] Connecting to VPS at {self.vps_url}...")

                self.websocket = await websockets.connect(
                    self.vps_url,
                    ping_interval=20,
                    ping_timeout=180,
                    close_timeout=1
                )

                self.connected = True
                print("[CLIENT] Connected to VPS!")

                # Send connect message
                connect_msg = WebSocketMessage(
                    type=MessageType.CONNECT,
                    payload={
                        "client_type": "local_spinal_cord",
                        "version": "1.0.0",
                        "timestamp": time.time()
                    }
                )
                await self.send_message(connect_msg)

                # Start background tasks
                self._listener_task = asyncio.create_task(self._listen())
                self._heartbeat_task = asyncio.create_task(self._heartbeat())

                # Notify connected
                if self.on_connect:
                    self.on_connect()

                # Stay connected until disconnected
                await self._wait_for_disconnect()

            except ConnectionRefusedError:
                print(f"[CLIENT] Connection refused. Retrying in {self.reconnect_interval}s...")
            except ConnectionClosedError as e:
                print(f"[CLIENT] Connection closed: {e}")
            except Exception as e:
                print(f"[CLIENT] Connection error: {e}")

            # Cleanup and retry
            self.connected = False
            await self._cleanup()
            await asyncio.sleep(self.reconnect_interval)

    async def _wait_for_disconnect(self):
        """Wait until connection is closed."""
        if self.websocket:
            try:
                await self.websocket.wait_closed()
            except:
                pass

    async def _listen(self):
        """Listen for incoming messages from VPS."""
        try:
            async for message in self.websocket:
                try:
                    ws_msg = WebSocketMessage.from_json(message)
                    await self._handle_message(ws_msg)
                except json.JSONDecodeError as e:
                    print(f"[CLIENT] Failed to parse message: {e}")
                except Exception as e:
                    print(f"[CLIENT] Error handling message: {e}")

        except ConnectionClosed:
            print("[CLIENT] Listener disconnected")
        except Exception as e:
            print(f"[CLIENT] Listener error: {e}")

    async def _handle_message(self, message: WebSocketMessage):
        """
        Handle incoming message from VPS.

        Args:
            message: Received WebSocket message.
        """
        msg_type = message.type
        payload = message.payload

        if msg_type == MessageType.PONG:
            # Heartbeat response
            pass

        elif msg_type == MessageType.STRATEGY_UPDATE:
            # Received new strategy policy
            try:
                self.latest_strategy = StrategyPolicy(**payload)
                print(f"[CLIENT] New strategy: {self.latest_strategy.action}")

                if self.on_strategy_update:
                    self.on_strategy_update(self.latest_strategy)
            except Exception as e:
                print(f"[CLIENT] Failed to parse strategy: {e}")

        elif msg_type == MessageType.POLICY_CONFIRMATION:
            # Confirm strategy was received/acknowledged
            print(f"[CLIENT] Policy confirmed: {payload.get('policy_id')}")

        elif msg_type == MessageType.DATABASE_QUERY:
            # Received query request from VPS
            print(f"[CLIENT] Database query: {payload.get('query_type')}")

            if self.on_database_query:
                self.on_database_query(payload)

        elif msg_type == MessageType.PING:
            # Respond to ping
            pong_msg = WebSocketMessage(
                type=MessageType.PONG,
                payload={"timestamp": time.time()}
            )
            await self.send_message(pong_msg)

        else:
            print(f"[CLIENT] Unknown message type: {msg_type}")

    async def _heartbeat(self):
        """Send periodic heartbeat to keep connection alive."""
        while self.connected:
            try:
                heartbeat_msg = WebSocketMessage(
                    type=MessageType.HEARTBEAT,
                    payload={
                        "timestamp": time.time(),
                        "status": "alive"
                    }
                )
                await self.send_message(heartbeat_msg)
            except Exception as e:
                print(f"[CLIENT] Heartbeat error: {e}")
                break

            await asyncio.sleep(self.heartbeat_interval)

    async def send_message(self, message: WebSocketMessage):
        """
        Send message to VPS.

        Args:
            message: WebSocket message to send.
        """
        if self.websocket and self.connected:
            try:
                await self.websocket.send(message.to_json())
            except ConnectionClosed:
                print("[CLIENT] Cannot send - connection closed")
                raise
            except Exception as e:
                print(f"[CLIENT] Send error: {e}")
                raise

    async def send_state_update(self, state: LocalState):
        """
        Send local state update to VPS.

        Args:
            state: Current local state.
        """
        msg = WebSocketMessage(
            type=MessageType.STATE_UPDATE,
            payload=state.to_dict()
        )
        await self.send_message(msg)

    async def send_database_response(self, query_id: str, response_data: Any):
        """
        Send response to database query from VPS.

        Args:
            query_id: ID of the query being responded to.
            response_data: Response data.
        """
        msg = WebSocketMessage(
            type=MessageType.DATABASE_RESPONSE,
            payload={
                "query_id": query_id,
                "data": response_data,
                "timestamp": time.time()
            }
        )
        await self.send_message(msg)

    async def send_error_report(self, error: str, context: Optional[Dict] = None):
        """
        Send error report to VPS.

        Args:
            error: Error message.
            context: Additional context about the error.
        """
        msg = WebSocketMessage(
            type=MessageType.ERROR_REPORT,
            payload={
                "error": error,
                "context": context or {},
                "timestamp": time.time()
            }
        )
        await self.send_message(msg)

    async def _cleanup(self):
        """Clean up resources."""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None

        # Notify disconnected
        if self.on_disconnect:
            self.on_disconnect()

    async def stop(self):
        """Stop the client."""
        self._running = False

        # Send disconnect message
        if self.connected:
            try:
                disconnect_msg = WebSocketMessage(
                    type=MessageType.DISCONNECT,
                    payload={"reason": "shutdown"}
                )
                await self.send_message(disconnect_msg)
            except:
                pass

        await self._cleanup()

    def start(self):
        """Start the client (synchronous wrapper for async)."""
        self._running = True

        # Run in background thread or with asyncio
        try:
            asyncio.run(self.connect())
        except KeyboardInterrupt:
            print("[CLIENT] Stopping...")
            asyncio.run(self.stop())

    async def start_async(self):
        """Start the client (async method)."""
        self._running = True
        await self.connect()

    def set_strategy_handler(self, handler: Callable[[StrategyPolicy], None]):
        """
        Set callback for strategy updates.

        Args:
            handler: Function to call when strategy updates received.
        """
        self.on_strategy_update = handler

    def set_connect_handler(self, handler: Callable[[], None]):
        """
        Set callback for connection established.

        Args:
            handler: Function to call when connected.
        """
        self.on_connect = handler

    def set_disconnect_handler(self, handler: Callable[[], None]):
        """
        Set callback for connection lost.

        Args:
            handler: Function to call when disconnected.
        """
        self.on_disconnect = handler

    def get_latest_strategy(self) -> Optional[StrategyPolicy]:
        """Get the most recently received strategy policy."""
        return self.latest_strategy

    def is_connected(self) -> bool:
        """Check if connected to VPS."""
        return self.connected


class VPSClientSync:
    """
    Synchronous wrapper around VPSClient for easier integration.
    """

    def __init__(
        self,
        vps_url: str = "ws://167.86.105.39:8000/ws",
        reconnect_interval: float = 5.0,
        heartbeat_interval: float = 30.0
    ):
        """
        Initialize sync wrapper.

        Args:
            vps_url: WebSocket URL of VPS server.
            reconnect_interval: Seconds between reconnection attempts.
            heartbeat_interval: Seconds between heartbeat messages.
        """
        self.client = VPSClient(vps_url, reconnect_interval, heartbeat_interval)

        # Event loop for async operations in separate thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._thread = None
        self._stop_event = asyncio.Event()

    def _run_event_loop(self):
        """Run event loop in background thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.client.start_async())

    def start(self):
        """Start the client in background thread."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self._thread.start()
            print("[CLIENT-SYNC] Started in background thread")

    def stop(self):
        """Stop the client."""
        self.client._running = False
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.client.stop())
        if self._thread:
            self._thread.join(timeout=5)

    def send_state_update(self, state: LocalState):
        """
        Send state update (synchronous wrapper).

        Args:
            state: Local state to send.
        """
        asyncio.run_coroutine_threadsafe(
            self.client.send_state_update(state),
            self.loop
        )

    def get_latest_strategy(self) -> Optional[StrategyPolicy]:
        """Get latest strategy (synchronous)."""
        return self.client.get_latest_strategy()

    def is_connected(self) -> bool:
        """Check if connected (synchronous)."""
        return self.client.is_connected()

    # Delegate other methods
    def set_strategy_handler(self, handler):
        self.client.set_strategy_handler(handler)

    def set_connect_handler(self, handler):
        self.client.set_connect_handler(handler)

    def set_disconnect_handler(self, handler):
        self.client.set_disconnect_handler(handler)


# Import for sync version
import threading
