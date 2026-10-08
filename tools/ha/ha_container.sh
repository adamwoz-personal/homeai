#!/usr/bin/env bash
# Manage the Home Assistant container (device backend for Jarvis).
# See plans/HOME_ASSISTANT_PLAN.md.
#
#   tools/ha/ha_container.sh start    # create (first time) or start
#   tools/ha/ha_container.sh status   # container state + HTTP check
#   tools/ha/ha_container.sh update   # pull new stable image and recreate
#   tools/ha/ha_container.sh logs     # follow logs
#   tools/ha/ha_container.sh stop
#
# Host networking is required: Hue bridge and Echo discovery use mDNS/SSDP.
# Config lives in $HA_CONFIG (default ~/homeassistant-config) and survives
# container recreation. restart=unless-stopped brings it back after reboot.
set -euo pipefail

NAME=${HA_NAME:-homeassistant}
IMAGE=${HA_IMAGE:-ghcr.io/home-assistant/home-assistant:stable}
CONFIG=${HA_CONFIG:-$HOME/homeassistant-config}
URL=${HA_URL:-http://127.0.0.1:8123}

# Works before the user's session has picked up the docker group.
docker_() {
    if docker info >/dev/null 2>&1; then docker "$@"
    else sg docker -c "$(printf '%q ' docker "$@")"; fi
}

exists() { docker_ container inspect "$NAME" >/dev/null 2>&1; }

create() {
    mkdir -p "$CONFIG"
    docker_ run -d --name "$NAME" \
        --restart unless-stopped \
        --network host \
        -e TZ="$(timedatectl show -p Timezone --value 2>/dev/null || echo UTC)" \
        -v "$CONFIG":/config \
        -v /run/dbus:/run/dbus:ro \
        "$IMAGE"
}

wait_http() {
    for _ in $(seq 1 90); do
        if curl -fs -o /dev/null "$URL/manifest.json"; then echo "HA is up at $URL"; return 0; fi
        sleep 2
    done
    echo "HA did not answer at $URL within 180 s; check: $0 logs" >&2
    return 1
}

case "${1:-status}" in
    start)
        if exists; then docker_ start "$NAME" >/dev/null; else docker_ pull "$IMAGE" && create; fi
        wait_http ;;
    stop)   docker_ stop "$NAME" ;;
    update)
        docker_ pull "$IMAGE"
        if exists; then docker_ rm -f "$NAME" >/dev/null; fi
        create && wait_http ;;
    logs)   docker_ logs -f --tail 100 "$NAME" ;;
    status)
        if ! exists; then echo "container $NAME: not created"; exit 1; fi
        docker_ inspect -f 'container {{.Name}}: {{.State.Status}} (image {{.Config.Image}}, restart {{.HostConfig.RestartPolicy.Name}})' "$NAME"
        if curl -fs -o /dev/null "$URL/manifest.json"; then echo "http: ok ($URL)"; else echo "http: not answering ($URL)"; exit 1; fi ;;
    *) echo "usage: $0 start|stop|update|logs|status" >&2; exit 2 ;;
esac
