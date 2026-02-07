#!/bin/bash
# Reflex-Strategy VPS Management Script

SERVICE_NAME="reflex-strategy-vps"
VPS_URL="http://localhost:8001"
EXTERNAL_URL="http://167.86.105.39:8001"
LOG_FILE="/var/log/reflex-strategy-vps.log"
ERROR_LOG="/var/log/reflex-strategy-vps-error.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_service() {
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${GREEN}✓ Service is running${NC}"
        return 0
    else
        echo -e "${RED}✗ Service is not running${NC}"
        return 1
    fi
}

status() {
    echo "=== Reflex-Strategy VPS Status ==="
    systemctl status "$SERVICE_NAME" --no-pager -l
}

start() {
    echo "Starting $SERVICE_NAME..."
    systemctl start "$SERVICE_NAME"
    sleep 2
    check_service
}

stop() {
    echo "Stopping $SERVICE_NAME..."
    systemctl stop "$SERVICE_NAME"
    echo "Service stopped."
}

restart() {
    echo "Restarting $SERVICE_NAME..."
    systemctl restart "$SERVICE_NAME"
    sleep 3
    check_service
}

logs() {
    echo "=== Recent Logs ==="
    tail -n 50 "$LOG_FILE"
}

error_logs() {
    echo "=== Error Logs ==="
    tail -n 50 "$ERROR_LOG"
}

journal() {
    echo "=== Systemd Journal Logs ==="
    journalctl -u "$SERVICE_NAME" -n 50 --no-pager
}

test_api() {
    echo "=== Testing API Endpoints ==="

    echo -e "\n${YELLOW}Health:${NC}"
    curl -s "$VPS_URL/health" | jq '.' 2>/dev/null || curl -s "$VPS_URL/health"

    echo -e "\n${YELLOW}Stats:${NC}"
    curl -s "$VPS_URL/stats" | jq '.' 2>/dev/null || curl -s "$VPS_URL/stats"

    echo -e "\n${YELLOW}Find nearest market:${NC}"
    curl -s "$VPS_URL/db/nearest?x=0&y=0&z=0&object_type=market" | jq '.' 2>/dev/null || curl -s "$VPS_URL/db/nearest?x=0&y=0&z=0&object_type=market"
}

db_query() {
    echo "=== Database Query ==="
    docker exec supabase-db psql -U postgres -d reflex_strategy -c "$1"
}

show_info() {
    echo "=== Reflex-Strategy VPS Information ==="
    echo -e "\n${GREEN}Service:${NC} $SERVICE_NAME"
    echo -e "${GREEN}Local URL:${NC} $VPS_URL"
    echo -e "${GREEN}External URL:${NC} $EXTERNAL_URL"
    echo -e "${GREEN}WebSocket:${NC} ws://167.86.105.39:8001/ws"
    echo -e "${GREEN}Log File:${NC} $LOG_FILE"
    echo -e "\n${YELLOW}Database:${NC}"
    db_query "SELECT 'Total entries: ' || COUNT(*) as count FROM spatial_memory;"
}

enable() {
    echo "Enabling $SERVICE_NAME to start on boot..."
    systemctl enable "$SERVICE_NAME"
}

disable() {
    echo "Disabling $SERVICE_NAME from starting on boot..."
    systemctl disable "$SERVICE_NAME"
}

help() {
    cat << EOF
Reflex-Strategy VPS Management Script

Usage: $0 [COMMAND]

Commands:
    start       - Start the VPS service
    stop        - Stop the VPS service
    restart     - Restart the VPS service
    status      - Show service status
    logs        - Show recent application logs
    journal     - Show systemd journal logs
    errors      - Show error logs
    test        - Test API endpoints
    db          - Run database query (use: db 'QUERY')
    info        - Show system information
    enable      - Enable service to start on boot
    disable     - Disable service from starting on boot
    help        - Show this help message

Examples:
    $0 status
    $0 test
    $0 restart
    $0 db "SELECT * FROM spatial_memory LIMIT 5;"
EOF
}

# Parse command
case "${1:-help}" in
    start)      start ;;
    stop)       stop ;;
    restart)    restart ;;
    status)     status ;;
    logs)       logs ;;
    journal)    journal ;;
    errors)     error_logs ;;
    test)       test_api ;;
    db)         db_query "$2" ;;
    info)       show_info ;;
    enable)     enable ;;
    disable)    disable ;;
    help)       help ;;
    *)          help ;;
esac
