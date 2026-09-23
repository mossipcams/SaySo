#!/usr/bin/env bash
# Self-check for gpu with vLLM stubbed out: never touches docker or the real GPU state.
# Run on the llm VM (needs flock): bash scripts/llm-host/test_gpu.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin"
cat > "$tmp/bin/docker" <<'DOCKER'
#!/usr/bin/env bash
case $1 in
  exec) shift 2; exec "$@" ;;
  restart) echo restart >> "$GPU_RUN_DIR/docker.log"; exit "${GPU_TEST_RESTART_RC:-0}" ;;
  *) exit 1 ;;
esac
DOCKER
chmod +x "$tmp/bin/docker"
: > "$tmp/docker.log"
export PATH="$tmp/bin:$PATH"
export GPU_RUN_DIR=$tmp GPU_VLLM_UP="echo up >> $tmp/vllm.log" GPU_VLLM_DOWN="echo down >> $tmp/vllm.log"
gpu() { "$here/gpu" "$@"; }
fail() { echo "FAIL: $*"; exit 1; }

gpu serve; [[ $(cat "$tmp/gpu.state") == serve && ! -e $tmp/locks/rocm.state ]] || fail "serve state"

gpu train wake sleep 3 & t=$!
sleep 1
[[ $(gpu status) == "state: train:wake"* ]] || fail "train state while running"
[[ ! -e $tmp/locks/rocm.state ]] || fail "legacy state must stay absent during training"
gpu serve 2>/dev/null && fail "serve must refuse while training"
gpu train lfm true 2>/dev/null && fail "second train must refuse"
wait $t
[[ $(gpu status) == "state: idle"* ]] || fail "idle after train"

echo "train:wake 999999" > "$tmp/gpu.state"             # dead pid
gpu serve || fail "dead train pid must count as idle"

gpu train wake --serve-after false 2>/dev/null && fail "failing cmd must fail"
[[ $(cat "$tmp/gpu.state") == serve ]] || fail "--serve-after must restart vLLM even when the run fails"

grep -q down "$tmp/vllm.log" || fail "train must stop vLLM"
# Cancellation must reap the worker before releasing the state.
"$here/gpu" train wake sleep 30 & t=$!
for _ in {1..100}; do
  read -r state owner worker < "$tmp/gpu.state"
  [[ $state == train:wake && -n ${worker:-} ]] && break
  sleep 0.05
done
[[ ${owner:-} == "$t" && -n ${worker:-} ]] || fail "worker reservation"
gpu stop
[[ $(gpu status) == "state: train:wake"* ]] || fail "stop must preserve training reservation"
kill -TERM "$t"
rc=0; wait "$t" || rc=$?
[[ $rc == 143 ]] || fail "cancellation exit code"
kill -0 "$worker" 2>/dev/null && fail "cancelled worker survived"
[[ $(gpu status) == "state: idle"* ]] || fail "idle after cancellation"

# A hard-killed wrapper must not release a live worker's reservation.
"$here/gpu" train wake sleep 30 & t=$!
for _ in {1..100}; do
  read -r state owner worker < "$tmp/gpu.state"
  [[ $state == train:wake && -n ${worker:-} ]] && break
  sleep 0.05
done
[[ ${owner:-} == "$t" && -n ${worker:-} ]] || fail "hard-kill worker reservation"
kill -KILL "$t"; wait "$t" 2>/dev/null || true
gpu serve 2>/dev/null && fail "orphan worker must block serving"
[[ $(gpu status) == *"owner alive: no"*"worker alive: yes"* ]] || fail "orphan status"
kill -TERM -- "-$worker"
for _ in {1..100}; do
  [[ $(gpu status) == "state: idle"* ]] && break
  sleep 0.05
done
[[ $(gpu status) == "state: idle"* ]] || fail "idle after orphan exits"

# A dead group leader does not mean its descendants have stopped.
"$here/gpu" train wake sh -c 'sleep 30 & echo $! > "$1"; wait' sh "$tmp/descendant" & t=$!
for _ in {1..100}; do
  read -r state owner worker < "$tmp/gpu.state"
  [[ $state == train:wake && -n ${worker:-} && -s $tmp/descendant ]] && break
  sleep 0.05
done
[[ ${owner:-} == "$t" && -n ${worker:-} && -s $tmp/descendant ]] || fail "descendant reservation"
kill -KILL "$t"; wait "$t" 2>/dev/null || true
kill -KILL "$worker"
sleep 0.1
kill -0 "$(cat "$tmp/descendant")" || fail "regression requires a live descendant"
gpu serve 2>/dev/null && fail "live descendant must block serving"
gpu train wake true 2>/dev/null && fail "live descendant must block training"
[[ $(gpu status) == *"worker group alive: yes"* ]] || fail "descendant status"
kill -TERM -- "-$worker"
for _ in {1..100}; do
  [[ $(gpu status) == "state: idle"* ]] && break
  sleep 0.05
done
[[ $(gpu status) == "state: idle"* ]] || fail "idle only after descendants stop"

# Dead Docker clients cannot prove the container command has stopped.
for action in serve stop train; do
  before=$(wc -l < "$tmp/docker.log" 2>/dev/null || echo 0)
  echo "train:lfm 2147483647 2147483646" > "$tmp/gpu.state"
  [[ $(gpu status) == "state: orphan:lfm"* ]] || fail "stale LFM must stay reserved"
  after=$(wc -l < "$tmp/docker.log" 2>/dev/null || echo 0)
  [[ $before == "$after" ]] || fail "status must not restart the container"
  if [[ $action == train ]]; then gpu train wake true; else gpu "$action"; fi
  after=$(wc -l < "$tmp/docker.log")
  [[ $after -eq $((before + 1)) ]] || fail "$action must recover stale LFM before proceeding"
done
echo "train:lfm 2147483647 2147483646" > "$tmp/gpu.state"
before=$(wc -l < "$tmp/vllm.log")
GPU_TEST_RESTART_RC=1 gpu serve 2>/dev/null && fail "failed stale LFM cleanup must block serving"
[[ $(wc -l < "$tmp/vllm.log") -eq $before ]] || fail "failed recovery must not start vLLM"
[[ $(cat "$tmp/gpu.state") == blocked:lfm-cleanup ]] || fail "failed recovery must stay blocked"
echo idle > "$tmp/gpu.state"

gpu train lfm true
[[ $(gpu status) == "state: idle"* ]] || fail "LFM success releases reservation"
gpu train lfm false 2>/dev/null && fail "LFM failure must fail"
grep -q restart "$tmp/docker.log" || fail "LFM failure must reap remote command"
[[ $(gpu status) == "state: idle"* ]] || fail "LFM failure releases after cleanup"
GPU_TEST_RESTART_RC=1 gpu train lfm false 2>/dev/null && fail "cleanup failure must fail"
gpu serve 2>/dev/null && fail "failed remote cleanup must block serving"
echo "starting:wake 2147483647" > "$tmp/gpu.state"
gpu serve 2>/dev/null && fail "incomplete launch must not release an unrecorded worker"
echo garbage > "$tmp/gpu.state"
gpu serve 2>/dev/null && fail "malformed state must fail closed"
echo idle > "$tmp/gpu.state"
echo "gpu self-check: ok"
