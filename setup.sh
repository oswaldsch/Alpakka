#!/usr/bin/env bash
#
# setup.sh — build, install and start alpakka.
#
# Installs the binary into PATH, writes a config pointing at the llama.cpp
# build found on this machine and a GGUF model directory, installs a systemd
# unit and starts it. Re-running updates everything in place.
#
#   ./setup.sh                    # user service, ~/.local/bin, ~/.config/alpakka
#   ./setup.sh --system           # system service, /usr/local/bin, /etc/alpakka
#   ./setup.sh --uninstall        # remove the service and the binary
#
# See --help for the full set of options.

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------- output ----

if [[ -t 1 ]]; then
	C_BOLD=$'\033[1m'; C_RED=$'\033[31m'; C_YELLOW=$'\033[33m'
	C_GREEN=$'\033[32m'; C_DIM=$'\033[2m'; C_OFF=$'\033[0m'
else
	C_BOLD=''; C_RED=''; C_YELLOW=''; C_GREEN=''; C_DIM=''; C_OFF=''
fi

step() { printf '%s==>%s %s\n' "$C_BOLD" "$C_OFF" "$*"; }
info() { printf '    %s\n' "$*"; }
warn() { printf '%swarning:%s %s\n' "$C_YELLOW" "$C_OFF" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$C_RED" "$C_OFF" "$*" >&2; exit 1; }

# run executes a command, echoing it under --dry-run instead.
run() {
	if (( DRY_RUN )); then
		printf '%s    would run: %s%s\n' "$C_DIM" "$*" "$C_OFF"
		return 0
	fi
	"$@"
}

# ----------------------------------------------------------------- flags ----

SCOPE=user
PREFIX=''
CONFIG_PATH=''
LISTEN=''
MODELS_ROOT=''
LIB_DIR=''
BACKEND=''
SERVICE_USER=''
DO_BUILD=1
DO_START=1
FORCE_CONFIG=0
CHECK_LLAMA=1
UNINSTALL=0
PURGE=0
DRY_RUN=0

usage() {
	cat <<'USAGE'
setup.sh — install alpakka as a systemd service.

Options:
  --user               install for the current user (default)
                       binary in ~/.local/bin, config in ~/.config/alpakka,
                       unit in ~/.config/systemd/user
  --system             install system-wide (uses sudo when not root)
                       binary in /usr/local/bin, config in /etc/alpakka,
                       unit in /etc/systemd/system
  --prefix DIR         directory to install the binary into
  --config PATH        config file to write and point the unit at
  --listen ADDR        listen address, e.g. 127.0.0.1:11435
  --models DIR         GGUF model directory (default: ~/models of the service user)
  --lib-dir DIR        directory holding llama-server (default: autodetected)
  --backend NAME       ggml backend subdirectory, e.g. rocm_v7_2, vulkan
  --service-user USER  user the system service runs as (default: invoking user)
  --force-config       overwrite an existing config (the old one is backed up)
  --skip-llama-check   do not verify llama-server supports the flags alpakka
                       passes; the failures then surface per request instead
  --no-build           install the ./alpakka binary already in the repo
  --no-start           install everything but do not enable or start the service
  --uninstall          stop and remove the service and the installed binary
  --purge              with --uninstall, also delete the config
  --dry-run            print what would be done, change nothing
  -h, --help           this text
USAGE
}

while (( $# )); do
	case "$1" in
		--user)          SCOPE=user ;;
		--system)        SCOPE=system ;;
		--prefix)        PREFIX="${2:?--prefix needs a directory}"; shift ;;
		--config)        CONFIG_PATH="${2:?--config needs a path}"; shift ;;
		--listen)        LISTEN="${2:?--listen needs an address}"; shift ;;
		--models)        MODELS_ROOT="${2:?--models needs a directory}"; shift ;;
		--lib-dir)       LIB_DIR="${2:?--lib-dir needs a directory}"; shift ;;
		--backend)       BACKEND="${2:?--backend needs a name}"; shift ;;
		--service-user)  SERVICE_USER="${2:?--service-user needs a name}"; shift ;;
		--force-config)  FORCE_CONFIG=1 ;;
		--skip-llama-check) CHECK_LLAMA=0 ;;
		--no-build)      DO_BUILD=0 ;;
		--no-start)      DO_START=0 ;;
		--uninstall)     UNINSTALL=1 ;;
		--purge)         PURGE=1 ;;
		--dry-run)       DRY_RUN=1 ;;
		-h|--help)       usage; exit 0 ;;
		*)               usage >&2; die "unknown option: $1" ;;
	esac
	shift
done

# --------------------------------------------------------------- layout -----

command -v systemctl >/dev/null || die "systemd is required (systemctl not found)"

if [[ $SCOPE == system ]]; then
	if (( EUID == 0 )); then
		SUDO=()
	elif command -v sudo >/dev/null; then
		SUDO=(sudo)
	else
		die "--system needs root and sudo is not installed"
	fi
	SYSTEMCTL=("${SUDO[@]}" systemctl)
	UNIT_DIR=/etc/systemd/system
	: "${PREFIX:=/usr/local/bin}"
	: "${CONFIG_PATH:=/etc/alpakka/config.toml}"
	: "${SERVICE_USER:=${SUDO_USER:-$USER}}"
else
	SUDO=()
	SYSTEMCTL=(systemctl --user)
	UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
	: "${PREFIX:=$HOME/.local/bin}"
	: "${CONFIG_PATH:=${XDG_CONFIG_HOME:-$HOME/.config}/alpakka/config.toml}"
	SERVICE_USER="$USER"
	[[ -n ${DBUS_SESSION_BUS_ADDRESS:-} || -n ${XDG_RUNTIME_DIR:-} ]] ||
		warn "no user session bus; systemctl --user may not work from here"
fi

if [[ $SCOPE == user ]]; then JOURNAL="journalctl --user"; else JOURNAL="journalctl"; fi
UNIT_PATH="$UNIT_DIR/alpakka.service"
BIN_PATH="$PREFIX/alpakka"

# ------------------------------------------------------------ uninstall -----

if (( UNINSTALL )); then
	step "Removing the alpakka service"
	run "${SYSTEMCTL[@]}" disable --now alpakka.service 2>/dev/null || true
	[[ -e $UNIT_PATH ]] && run "${SUDO[@]}" rm -f "$UNIT_PATH"
	run "${SYSTEMCTL[@]}" daemon-reload
	[[ -e $BIN_PATH ]] && run "${SUDO[@]}" rm -f "$BIN_PATH"
	if (( PURGE )) && [[ -e $CONFIG_PATH ]]; then
		run "${SUDO[@]}" rm -f "$CONFIG_PATH"
		info "removed $CONFIG_PATH"
	elif [[ -e $CONFIG_PATH ]]; then
		info "kept $CONFIG_PATH (use --purge to remove it)"
	fi
	step "Done."
	exit 0
fi

# --------------------------------------------------------------- detect -----

# detect_lib_dir finds the directory holding llama-server.
detect_lib_dir() {
	local c p
	for c in /usr/local/lib/llama.cpp /opt/llama.cpp/bin /usr/local/bin; do
		[[ -x $c/llama-server ]] && { printf '%s\n' "$c"; return 0; }
	done
	if p="$(command -v llama-server 2>/dev/null)"; then
		p="$(readlink -f "$p")"
		printf '%s\n' "$(dirname "$p")"
		return 0
	fi
	return 1
}

# detect_backend picks the ggml backend subdirectory of lib_dir. The choice is
# load-bearing: llama-server runs with this directory as its working directory,
# because ggml looks for libggml-hip.so there and starts silently on CPU if it
# is missing.
detect_backend() {
	local lib="$1" d name avail=() pref=()
	for d in "$lib"/*/; do
		name="$(basename "$d")"
		compgen -G "$d/libggml-*.so*" >/dev/null 2>&1 && avail+=("$name")
	done
	if (( ! ${#avail[@]} )); then
		# A hand-built llama.cpp keeps its libraries beside the binary, and a
		# distro package links against them in the system library path. Either
		# way there is no backend directory to enter, and "." — the binary's own
		# directory — is the harmless answer alpakka's cd needs.
		compgen -G "$lib/libggml-*.so*" >/dev/null 2>&1 && { printf '.\n'; return 0; }
		if ldd "$lib/llama-server" 2>/dev/null | grep -q 'libggml.*=> */'; then
			printf '.\n'; return 0
		fi
		return 1
	fi

	if [[ -e /dev/kfd ]]; then
		pref=(rocm hip)
	elif [[ -e /dev/nvidia0 ]] || command -v nvidia-smi >/dev/null 2>&1; then
		pref=(cuda)
	fi
	pref+=(vulkan metal cpu)

	local want match
	for want in "${pref[@]}"; do
		# Highest version wins, so rocm_v7_2 beats rocm_v6_3.
		match="$(printf '%s\n' "${avail[@]}" | grep -i "^${want}" | sort -V | tail -n1 || true)"
		[[ -n $match ]] && { printf '%s\n' "$match"; return 0; }
	done
	printf '%s\n' "${avail[0]}"
}

# AS_USER runs a command as the user the service will run as. Testing access as
# the invoking user instead is how a system service ends up installed, active,
# and serving nothing.
if [[ $SERVICE_USER == "$USER" ]]; then
	AS_USER=()
elif (( EUID == 0 )) && command -v runuser >/dev/null; then
	AS_USER=(runuser -u "$SERVICE_USER" --)
elif command -v sudo >/dev/null; then
	AS_USER=(sudo -n -u "$SERVICE_USER" --)
else
	AS_USER=()  # cannot drop privileges; the checks below test as us instead
fi

# readable_as reports whether the service user can read path, with the tests
# systemd's own view would apply: a directory also needs the traverse bit, a
# device node does not.
readable_as() {
	local path="$1" args=(-r "$1")
	[[ -d $path ]] && args+=(-a -x "$path")
	if (( ${#AS_USER[@]} )); then
		"${AS_USER[@]}" test "${args[@]}" 2>/dev/null
	else
		test "${args[@]}"
	fi
}

step "Looking for the pieces"

if [[ -z $LIB_DIR ]]; then
	LIB_DIR="$(detect_lib_dir)" || die \
"no llama-server found.

Looked in /usr/local/lib/llama.cpp, /opt/llama.cpp/bin, /usr/local/bin and
on PATH. alpakka needs one from llama.cpp itself, recent enough for --fit and
--spec-type. Build it, or install a package that carries it, then re-run this
script (--lib-dir points at the directory holding the binary)."
fi
[[ -x $LIB_DIR/llama-server ]] || die "no executable llama-server in $LIB_DIR"
info "llama-server:  $LIB_DIR/llama-server"

if [[ -z $BACKEND ]]; then
	BACKEND="$(detect_backend "$LIB_DIR")" || die \
"no ggml backend directory under $LIB_DIR.

alpakka runs llama-server from the backend directory so ggml finds its
libraries. Pass one with --backend."
fi
[[ -d $LIB_DIR/$BACKEND ]] || die "backend directory $LIB_DIR/$BACKEND does not exist"
info "backend:       $BACKEND"

# alpakka's promises are made of specific llama-server flags: "--fit off" is
# what stops a silent CPU spill, and --spec-type is the speculation that beats
# the memory-bandwidth wall. An older build starts fine and then rejects every
# load with "invalid argument", which is a much worse place to find out. Distro
# packages are routinely a year behind, so this is worth a second here.
if (( CHECK_LLAMA )); then
	llama_help="$(timeout 20 "$LIB_DIR/llama-server" --help 2>&1 || true)"
	if [[ -z $llama_help ]]; then
		die "$LIB_DIR/llama-server printed nothing for --help, so it would not run.
Check it by hand, or pass --skip-llama-check to install anyway."
	fi
	missing=()
	for flag in --fit --spec-type --jinja --no-webui --no-mmproj; do
		grep -q -- "$flag" <<<"$llama_help" || missing+=("$flag")
	done
	# --flash-attn has to take on/off/auto; the older boolean form rejects the
	# value alpakka passes.
	if ! grep -- '--flash-attn' <<<"$llama_help" | grep -qE 'on|off|auto'; then
		missing+=('--flash-attn on|off|auto')
	fi
	if (( ${#missing[@]} )); then
		die "$LIB_DIR/llama-server is too old for alpakka.

It does not support: ${missing[*]}

alpakka passes these on every load, so this build would reject every request.
Update llama.cpp, or pass --skip-llama-check to install against it anyway."
	fi
	info "flags:         --fit, --spec-type and --flash-attn on/off/auto present"
	# KV cache streaming is still out of tree, so its absence is not a problem
	# unless a profile asks for it. Reporting it here saves guessing why
	# kv_stream_arena_mib was rejected.
	if grep -q -- '--kv-stream-arena-mib' <<<"$llama_help"; then
		info "kv streaming:  --kv-stream-arena-mib present"
	else
		info "kv streaming:  --kv-stream-arena-mib absent (kv_stream_arena_mib unavailable)"
	fi
	# The MoE expert cache is an unmerged draft PR, so absent is the normal case.
	if grep -q -- '--moe-expert-cache' <<<"$llama_help"; then
		info "moe cache:     --moe-expert-cache present"
	else
		info "moe cache:     --moe-expert-cache absent (moe_expert_cache unavailable)"
	fi
	# --n-cpu-moe and --override-tensor are upstream, but old enough builds
	# predate them and the failure is a load that exits on an unknown flag.
	moe_missing=()
	for flag in --n-cpu-moe --override-tensor; do
		grep -q -- "$flag" <<<"$llama_help" || moe_missing+=("$flag")
	done
	if (( ${#moe_missing[@]} )); then
		info "moe offload:   ${moe_missing[*]} absent (num_cpu_moe/override_tensor unavailable)"
	else
		info "moe offload:   --n-cpu-moe and --override-tensor present"
	fi
fi

if [[ -z $MODELS_ROOT ]]; then
	MODELS_ROOT="$(getent passwd "$SERVICE_USER" | cut -d: -f6)/models"
fi
info "model root:    $MODELS_ROOT"

# alpakka refuses to start without a readable model root, so create it empty.
if [[ ! -d $MODELS_ROOT ]]; then
	run "${SUDO[@]}" install -d -m 0755 -o "$SERVICE_USER" "$MODELS_ROOT"
	info "created $MODELS_ROOT, add models with: alpakka pull <ref>"
elif ! readable_as "$MODELS_ROOT"; then
	warn "$SERVICE_USER cannot read $MODELS_ROOT, so the service will serve no models.
    Grant read access, for example:
        sudo setfacl -R -m u:$SERVICE_USER:rX $MODELS_ROOT
        sudo setfacl -d -m u:$SERVICE_USER:rX $MODELS_ROOT"
fi

# The GPU is reached through /dev/kfd and /dev/dri. A service that cannot open
# them still starts, on CPU, at a few tokens a second and says nothing about it.
for dev in /dev/kfd /dev/dri/renderD128; do
	[[ -e $dev ]] || continue
	if ! readable_as "$dev"; then
		warn "$SERVICE_USER cannot open $dev, so llama-server would fall back to CPU.
    Add them to the device's group and log back in:
        sudo usermod -aG render,video $SERVICE_USER"
		break
	fi
done

# ---------------------------------------------------------------- build -----

if (( DO_BUILD )); then
	command -v go >/dev/null || die "go toolchain not found; install Go or pass --no-build"
	step "Building alpakka"
	run env -C "$REPO_DIR" go build -o "$REPO_DIR/alpakka" ./cmd/alpakka
else
	[[ -x $REPO_DIR/alpakka ]] || die "--no-build given but $REPO_DIR/alpakka is not there"
	step "Using the existing $REPO_DIR/alpakka"
fi

# -------------------------------------------------------------- install -----

step "Installing the binary"
run "${SUDO[@]}" install -d -m 0755 "$PREFIX"
run "${SUDO[@]}" install -m 0755 "$REPO_DIR/alpakka" "$BIN_PATH"
info "$BIN_PATH"

case ":$PATH:" in
	*":$PREFIX:"*) ;;
	*) warn "$PREFIX is not on your PATH; add it to your shell profile to run alpakka by name" ;;
esac

# --------------------------------------------------------------- config -----

step "Writing the config"
CONFIG_DIR="$(dirname "$CONFIG_PATH")"
run "${SUDO[@]}" install -d -m 0755 "$CONFIG_DIR"

write_config() {
	local tmp
	tmp="$(mktemp)"
	cat >"$tmp" <<TOML
# alpakka — written by setup.sh on $(date -Iseconds). Yours to edit.

[server]
listen = "$LISTEN"

[store]
roots = ["$MODELS_ROOT"]

[llama]
lib_dir = "$LIB_DIR"
backend = "$BACKEND"

[defaults]
num_ctx = 32768
cache_type_k = "q8_0"
cache_type_v = "q8_0"
flash_attn = "on"
keep_alive = "5m"

# Per-model profiles hold the settings ollama cannot express. Name the section
# after the model as it appears in 'alpakka list'.
#
# [models."qwen3.8-27b:q3-k-xl"]
# # MTP speculation is the only lever that beats the card's memory bandwidth.
# spec_type = "draft-mtp"
# spec_draft_n_max = 2
# reasoning_effort = "low"
# # The vision projector reserves ~1.16 GB whether or not it is used.
# projector = false
TOML
	run "${SUDO[@]}" install -m 0644 "$tmp" "$CONFIG_PATH"
	rm -f "$tmp"
}

if [[ -e $CONFIG_PATH ]] && (( ! FORCE_CONFIG )); then
	info "keeping the existing $CONFIG_PATH (--force-config replaces it)"
	# The unit is about to be pointed at this file, so the values that actually
	# take effect are the ones in it, not the ones detected above.
	if [[ -r $CONFIG_PATH ]]; then
		existing_listen="$(sed -n 's/^[[:space:]]*listen[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "$CONFIG_PATH" | head -n1)"
		[[ -n $existing_listen ]] && LISTEN="$existing_listen"
	fi
	: "${LISTEN:=127.0.0.1:11435}"
else
	: "${LISTEN:=127.0.0.1:11435}"
	if [[ -e $CONFIG_PATH ]]; then
		run "${SUDO[@]}" cp -a "$CONFIG_PATH" "$CONFIG_PATH.bak"
		info "backed up the old config to $CONFIG_PATH.bak"
	fi
	write_config
	info "$CONFIG_PATH"
fi
info "listening on $LISTEN"

# Port 11434 is ollama's. Sharing it with a running ollama means whichever
# binds first wins and the other dies on start, which is worth saying now.
if [[ $LISTEN == *:11434 ]] && systemctl is-active --quiet ollama.service 2>/dev/null; then
	warn "ollama.service is running and holds port 11434; stop it (sudo systemctl disable --now ollama) or pick another --listen"
fi

# ----------------------------------------------------------------- unit -----

step "Installing the systemd unit"

unit_common() {
	cat <<UNIT
[Unit]
Description=alpakka — ollama-compatible API served by llama-server
Documentation=file://$REPO_DIR/README.md
After=network-online.target
Wants=network-online.target

[Service]
Type=exec
ExecStart=$BIN_PATH -config $CONFIG_PATH
Restart=on-failure
RestartSec=2
# alpakka stops its llama-server child itself on SIGTERM; signalling only the
# main process lets it do that, and the cgroup SIGKILL is the backstop for a
# child that would otherwise outlive it holding the whole card.
KillMode=mixed
TimeoutStopSec=30
UNIT
}

TMP_UNIT="$(mktemp)"
if [[ $SCOPE == system ]]; then
	# Naming a group that does not exist makes systemd refuse to start the
	# unit, so only the ones this machine actually has go in.
	gpu_groups=()
	for g in render video; do
		getent group "$g" >/dev/null && gpu_groups+=("$g")
	done
	{
		unit_common
		printf 'User=%s\n' "$SERVICE_USER"
		if (( ${#gpu_groups[@]} )); then
			echo "# The GPU is reached through /dev/kfd and /dev/dri, so these must be held."
			printf 'SupplementaryGroups=%s\n' "${gpu_groups[*]}"
		fi
		cat <<'UNIT'
NoNewPrivileges=yes
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
UNIT
	} >"$TMP_UNIT"
else
	{
		unit_common
		cat <<'UNIT'

[Install]
WantedBy=default.target
UNIT
	} >"$TMP_UNIT"
fi

run "${SUDO[@]}" install -d -m 0755 "$UNIT_DIR"
run "${SUDO[@]}" install -m 0644 "$TMP_UNIT" "$UNIT_PATH"
info "$UNIT_PATH"
if (( DRY_RUN )); then
	while IFS= read -r line; do
		printf '%s    | %s%s\n' "$C_DIM" "$line" "$C_OFF"
	done <"$TMP_UNIT"
fi
rm -f "$TMP_UNIT"

run "${SYSTEMCTL[@]}" daemon-reload

if (( ! DO_START )); then
	step "Installed, not started (--no-start)."
	info "start it with: ${SYSTEMCTL[*]} enable --now alpakka"
	exit 0
fi

# A user service is stopped when the last session ends unless lingering is on,
# which is not what anyone wants from a model server.
if [[ $SCOPE == user ]] && command -v loginctl >/dev/null; then
	if [[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != yes ]]; then
		if ! run loginctl enable-linger "$USER" 2>/dev/null; then
			warn "could not enable lingering; the service will stop when you log out.
    Fix with: sudo loginctl enable-linger $USER"
		fi
	fi
fi

step "Starting alpakka"
run "${SYSTEMCTL[@]}" enable alpakka.service
# restart rather than start, so re-running this script picks up the new binary.
run "${SYSTEMCTL[@]}" restart alpakka.service

# ---------------------------------------------------------------- check -----

if (( DRY_RUN )); then
	step "Dry run complete; nothing was changed."
	exit 0
fi

# Wait for the API rather than for the unit: an active unit whose port is not
# yet open is not something a client can use.
probe_host="${LISTEN%:*}"
probe_port="${LISTEN##*:}"
case "$probe_host" in ''|'0.0.0.0'|'::'|'[::]') probe_host=127.0.0.1 ;; esac

ready=0
for _ in $(seq 1 30); do
	if curl -fsS --max-time 2 "http://$probe_host:$probe_port/api/version" >/dev/null 2>&1; then
		ready=1
		break
	fi
	"${SYSTEMCTL[@]}" is-active --quiet alpakka.service || break
	sleep 0.5
done

if (( ! ready )); then
	warn "alpakka did not answer on http://$probe_host:$probe_port"
	"${SYSTEMCTL[@]}" --no-pager --lines=20 status alpakka.service || true
	die "see $JOURNAL -u alpakka -e for the reason"
fi

version="$(curl -fsS "http://$probe_host:$probe_port/api/version" 2>/dev/null || true)"
models="$(curl -fsS "http://$probe_host:$probe_port/api/tags" 2>/dev/null | grep -o '"name"' | wc -l || true)"

step "${C_GREEN}alpakka is up${C_OFF} on http://$probe_host:$probe_port"
info "version:  $version"
info "models:   $models found"
echo
info "point clients at it:"
info "    export OLLAMA_HOST=$probe_host:$probe_port"
info "    alpakka list"
echo
info "logs:     $JOURNAL -u alpakka -f"
info "restart:  ${SYSTEMCTL[*]} restart alpakka"
info "config:   $CONFIG_PATH"
