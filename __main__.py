#!/usr/bin/env python
"""
Main entry point for Reflex-Strategy System.
Run with: python -m reflex_strategy_system [command]
"""
import sys
import argparse
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Reflex-Strategy System - Hybrid automation architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m reflex_strategy_system local      # Run Local PC (Spinal Cord)
  python -m reflex_strategy_system vps        # Run VPS (Brain)
  python -m reflex_strategy_system setup-db   # Set up database
  python -m reflex_strategy_system --version  # Show version

Architecture:
  Local PC (Spinal Cord): Immediate reflexes and low-latency perception
  VPS (Brain): Long-term strategic planning using LLM and spatial memory
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Reflex-Strategy System 1.0.0"
    )

    parser.add_argument(
        "command",
        choices=["local", "vps", "debug", "setup-db", "setup-local-db"],
        nargs="?",
        help="Command to run"
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (for VPS)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (for VPS)"
    )

    parser.add_argument(
        "--db-host",
        default="localhost",
        help="Database host"
    )

    parser.add_argument(
        "--db-port",
        type=int,
        default=5432,
        help="Database port"
    )

    parser.add_argument(
        "--db-user",
        default="postgres",
        help="Database user"
    )

    parser.add_argument(
        "--db-name",
        default="reflex_strategy",
        help="Database name"
    )

    parser.add_argument(
        "--vps-url",
        default="ws://localhost:8000/ws",
        help="VPS WebSocket URL (for Local PC)"
    )

    args = parser.parse_args()

    # Load environment variables from .env if exists
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)

    # Execute command
    if args.command == "local":
        print("[SYSTEM] Starting Local PC (Spinal Cord)...")
        from local.coordinator import LocalCoordinator

        import asyncio

        # Override config with CLI args
        os.environ["VPS_URL"] = os.getenv("VPS_URL", args.vps_url)

        coordinator = LocalCoordinator()

        async def run_local():
            try:
                await coordinator.initialize()
                await coordinator.start()
            except KeyboardInterrupt:
                print("\n[SYSTEM] Shutting down Local PC...")
            finally:
                await coordinator.stop()

        asyncio.run(run_local())

    elif args.command == "debug":
        # Debug viewer - pass through any remaining args
        sys.argv = ["debug_view.py"] + sys.argv[2:]
        from local import debug_view
        debug_view.main()

    elif args.command == "vps":
        print("[SYSTEM] Starting VPS (Brain)...")
        from vps.server import VPSBrainServer

        # Override config
        os.environ["VPS_HOST"] = args.host
        os.environ["VPS_PORT"] = str(args.port)
        os.environ["DB_HOST"] = os.getenv("DB_HOST", args.db_host)
        os.environ["DB_PORT"] = os.getenv("DB_PORT", str(args.db_port))
        os.environ["DB_NAME"] = os.getenv("DB_NAME", args.db_name)
        os.environ["DB_USER"] = os.getenv("DB_USER", args.db_user)

        server = VPSBrainServer(
            host=args.host,
            port=args.port,
            db_host=os.getenv("DB_HOST", args.db_host),
            db_port=int(os.getenv("DB_PORT", str(args.db_port))),
            db_name=os.getenv("DB_NAME", args.db_name),
            db_user=os.getenv("DB_USER", args.db_user),
            db_password=os.getenv("DB_PASSWORD"),
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini")
        )

        # Run server
        import asyncio

        async def run_vps():
            await server.startup()
            server.run()

        asyncio.run(run_vps())

    elif args.command == "setup-local-db":
        print("[SYSTEM] Setting up database locally...")
        import asyncio
        from setup_local import setup_database

        db_password = os.getenv("DB_PASSWORD")
        if not db_password:
            db_password = input("Enter PostgreSQL password: ")

        asyncio.run(setup_database(
            host=os.getenv("DB_HOST", args.db_host),
            port=int(os.getenv("DB_PORT", str(args.db_port))),
            database=os.getenv("DB_NAME", args.db_name),
            user=os.getenv("DB_USER", args.db_user),
            password=db_password,
            create_db=True,
            setup_postgis=True
        ))

    elif args.command == "setup-db":
        print("[SYSTEM] The 'setup-db' command requires direct SQL execution.")
        print("[SYSTEM] Use 'python setup_local.py' or 'python -m reflex_strategy_system setup-local-db' instead.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
