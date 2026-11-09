# Sourced by install.sh: install Jarvis's service, user or system unit. (SANDBOX_PLAN)
#
# Why a separate file: the choice has four cases and a wrong one either leaves
# Jarvis unsandboxed without a word or runs two Jarvises on one microphone.
# tests/test_install_jarvis_unit.py sources this with fake run/info/note.
#
# Needs from the caller: REPO ZC_DIR UNIT (user unit path) SYSTEM_UNIT USE_SUDO
# RENDERED (a temp file), and the functions run, info, note.
# USERNS_SYSCTL is the AppArmor switch that makes a user unit's sandbox a no-op.
USERNS_SYSCTL="${USERNS_SYSCTL:-/proc/sys/kernel/apparmor_restrict_unprivileged_userns}"

user_sandbox_blocked() {   # true when a user unit's ProtectHome would silently do nothing
    [[ "$(cat "$USERNS_SYSCTL" 2>/dev/null)" == 1 ]]
}

install_jarvis_unit() {
    if [[ $USE_SUDO -eq 1 ]]; then
        # System unit (sandbox enforced). The script installs the polkit rule, moves any
        # user unit aside first, updates a changed unit and rolls back if it won't stay up.
        run sudo "$REPO/tools/install_system_service.sh"
        return
    fi
    if [[ -f "$SYSTEM_UNIT" ]]; then
        # Never bring the user unit back beside the system one.
        info "Jarvis runs as a system unit ($SYSTEM_UNIT); to update it re-run with --sudo, or: sudo tools/install_system_service.sh"
        return
    fi
    sed -e "s#@REPO@#${REPO}#g" -e "s#@ZEROCLAW_DIR@#${ZC_DIR}#g" "$REPO/deploy/systemd/homeai.service.in" > "$RENDERED"
    if [[ -f "$UNIT" ]] && cmp -s "$RENDERED" "$UNIT"; then
        info "service unit up to date"
    else
        if [[ -f "$UNIT" ]]; then run cp "$UNIT" "$UNIT.bak.$(date +%s)"; fi
        run install -D -m 644 "$RENDERED" "$UNIT"
        run systemctl --user daemon-reload
    fi
    run systemctl --user enable homeai.service
    if user_sandbox_blocked; then
        note "Jarvis's sandbox (read-only home) is NOT in force: this system blocks it for user services. Re-run with --sudo, or: sudo tools/install_system_service.sh"
    fi
}
