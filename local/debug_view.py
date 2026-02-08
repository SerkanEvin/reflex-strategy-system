#!/usr/bin/env python3
"""
Debug Mode Viewer CLI
View and analyze debug logs and sessions from the reflex-strategy system.
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))


def load_debug_log(log_file: str) -> List[Dict[str, Any]]:
    """Load debug log from JSONL file."""
    entries = []
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_file}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing log file: {e}")
        return []
    return entries


def load_session(session_file: str) -> Dict[str, Any]:
    """Load session JSON file."""
    try:
        with open(session_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Session file not found: {session_file}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing session file: {e}")
        return {}


def print_summary(entries: List[Dict[str, Any]], session: Dict[str, Any] = None):
    """Print summary of debug data."""
    print("=" * 80)
    print("Debug Session Summary")
    print("=" * 80)

    if session:
        stats = session.get("statistics", {})
        print(f"\nSession ID: {session.get('session_id', 'unknown')}")
        print(f"Duration: {(session.get('session_end', 0) - session.get('session_start', 0)):.1f}s")
        print(f"Total Events: {stats.get('total_events', 0):,}")
        print(f"Total Actions: {stats.get('total_actions', 0):,}")
        print(f"Detection Records: {stats.get('total_detections_records', 0):,}")

        # Action breakdown
        actions_by_type = {}
        for k, v in stats.items():
            if k.startswith('action_') and not k.startswith('action_type'):
                action_name = k[7:]  # Remove 'action_' prefix
                if action_name not in action_type_names:
                    actions_by_type[action_name] = v

        if actions_by_type:
            print("\nActions by Type:")
            for action, count in sorted(actions_by_type.items(), key=lambda x: -x[1])[:10]:
                print(f"  {action}: {count:,}")

    else:
        print(f"\nTotal Entries: {len(entries):,}")

        # Count by type
        type_counts: Dict[str, int] = {}
        for entry in entries:
            entry_type = entry.get("_type", entry.get("event_type", "unknown"))
            type_counts[entry_type] = type_counts.get(entry_type, 0) + 1

        print("\nEntries by Type:")
        for entry_type, count in sorted(type_counts.items()):
            print(f"  {entry_type}: {count:,}")

    print("=" * 80)


action_type_names = {"reflex", "strategy"}

def print_actions(entries: List[Dict[str, Any]], limit: int = 20):
    """Print action history."""
    actions = [e for e in entries if e.get("_type") == "action" or "action" in e.get("event_type", "")]
    actions.sort(key=lambda a: a.get("timestamp", 0), reverse=True)

    actions = actions[:limit]

    print(f"\n{'='*80}")
    print(f"Recent Actions (showing {len(actions)} of {len([e for e in entries if e.get('_type') == 'action'])} total)")
    print("=" * 80)

    for action in actions:
        timestamp = action.get("timestamp", 0)
        time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
        action_type = action.get("action_type", "unknown").upper()
        action_name = action.get("action", "unknown")
        reasoning = action.get("reasoning", "")[:60]
        print(f"[{time_str}] {action_type:8s} {action_name:15s} - {reasoning}")

    print("=" * 80)


def print_events(entries: List[Dict[str, Any]], event_type: str = None, limit: int = 20):
    """Print event log."""
    events = [e for e in entries if e.get("_type") != "action" and e.get("_type") != "detection"]

    if event_type:
        events = [e for e in events if e.get("event_type") == event_type]

    events.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
    events = events[:limit]

    print(f"\n{'='*80}")
    label = event_type or "Events"
    print(f"{label} (showing {len(events)})")
    print("=" * 80)

    for event in events:
        timestamp = event.get("timestamp", 0)
        time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
        source = event.get("source", "unknown")
        etype = event.get("event_type", "unknown")
        data = event.get("data", {})

        line = f"[{time_str}] {source:15s} {etype}"
        if isinstance(data, dict) and data:
            # Show first few data values
            data_str = ", ".join(f"{k}={v}" for k, v in list(data.items())[:3])
            line += f" | {data_str}"
        print(line)

    print("=" * 80)


def print_detections(entries: List[Dict[str, Any]], limit: int = 5):
    """Print detection records."""
    detections = [e for e in entries if e.get("_type") == "detection"]
    detections.sort(key=lambda d: d.get("timestamp", 0), reverse=True)
    detections = detections[:limit]

    print(f"\n{'='*80}")
    print(f"Recent Detection Records (showing {len(detections)})")
    print("=" * 80)

    for det in detections:
        timestamp = det.get("timestamp", 0)
        time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
        frame_id = det.get("frame_id", 0)
        detections_list = det.get("detections", [])
        security_tier = det.get("security_tier", "unknown")

        print(f"[{time_str}] Frame {frame_id} | Security: {security_tier}")
        for d in detections_list[:3]:
            label = d.get("label", "unknown")
            dtype = d.get("detection_type", "unknown")
            conf = d.get("confidence", 0)
            dist = d.get("distance")
            dist_str = f"{dist:.1f}m" if dist else "?"
            print(f"  - {label} ({dtype}) conf:{conf:.2f} dist:{dist_str}")

    print("=" * 80)


def list_available_logs():
    """List available debug log files."""
    logs_dir = Path(__file__).parent.parent / "logs"
    if not logs_dir.exists():
        print(f"No logs directory found: {logs_dir}")
        return

    print(f"\nAvailable files in {logs_dir}:")
    print("=" * 80)

    # Find log files
    log_files = sorted(logs_dir.glob("debug_*.jsonl"))
    session_files = sorted(logs_dir.glob("session_*.json"))

    if not log_files and not session_files:
        print("No debug log or session files found.")
        return

    print("\nDebug Logs (JSONL):")
    for f in log_files[-5:]:  # Show last 5
        size = f.stat().st_size / 1024
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {f.name:30s} {size:8.1f} KB | {mtime}")

    print("\nSession Exports (JSON):")
    for f in session_files[-5:]:
        size = f.stat().st_size / 1024
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {f.name:30s} {size:8.1f} KB | {mtime}")

    print("=" * 80)


def export_to_csv(entries: List[Dict[str, Any]], output_file: str):
    """Export debug data to CSV."""
    import csv

    if not entries:
        print("No entries to export.")
        return

    # Get all possible fields
    fieldnames = set()
    for entry in entries:
        fieldnames.update(entry.keys())

    fieldnames = sorted(fieldnames)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            # Flatten nested dicts for CSV
            flattened = {}
            for key, value in entry.items():
                if isinstance(value, (dict, list)):
                    flattened[key] = json.dumps(value)
                else:
                    flattened[key] = value
            writer.writerow(flattened)

    print(f"Exported {len(entries)} entries to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Debug Mode Viewer for Reflex-Strategy System")
    parser.add_argument("command", choices=["view", "list", "export"], help="Command to run")
    parser.add_argument("--file", "-f", help="Path to log or session file")
    parser.add_argument("--actions", "-a", action="store_true", help="Show action history")
    parser.add_argument("--events", "-e", help="Filter events by type")
    parser.add_argument("--detections", "-d", action="store_true", help="Show detection records")
    parser.add_argument("--summary", "-s", action="store_true", help="Show summary")
    parser.add_argument("--limit", "-l", type=int, default=20, help="Number of entries to show")
    parser.add_argument("--output", "-o", help="Output file for export command")
    parser.add_argument("--format", choices=["json", "csv"], default="json", help="Export format")

    args = parser.parse_args()

    if args.command == "list":
        list_available_logs()
        return

    # For view and export commands, need a file
    if args.command in ["view", "export"]:
        if args.file:
            # Use specified file
            if args.file.endswith(".json"):
                entries = load_session(args.file)
                session = entries
                entries = entries.get("events", []) + entries.get("actions", []) + entries.get("detections", [])
            else:
                entries = load_debug_log(args.file)
                session = None
        else:
            # Use latest log file
            logs_dir = Path(__file__).parent.parent / "logs"
            log_files = sorted(logs_dir.glob("debug_*.jsonl"))
            if not log_files:
                print("No log files found. Use --list to see available files.")
                return
            args.file = str(log_files[-1])
            print(f"Using latest log file: {args.file}")
            entries = load_debug_log(args.file)
            session = None

        if args.command == "export":
            if args.format == "csv":
                export_to_csv(entries, args.output or "debug_export.csv")
            else:
                # JSON export
                with open(args.output or "debug_export.json", "w", encoding="utf-8") as f:
                    json.dump(entries, f, indent=2, default=str)
                print(f"Exported {len(entries)} entries to {args.output or 'debug_export.json'}")
            return

        # View command
        if args.summary or not (args.actions or args.events or args.detections):
            print_summary(entries, session)

        if args.events:
            print_events(entries, args.events, args.limit)

        if args.actions:
            print_actions(entries, args.limit)

        if args.detections:
            print_detections(entries, args.limit)


if __name__ == "__main__":
    main()
