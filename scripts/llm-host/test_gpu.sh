#!/usr/bin/env bash
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
printf '#!/bin/sh\necho "VRAM%%: stub"\n' > "$tmp/bin/rocm-smi"
chmod +x "$tmp/bin/docker" "$tmp/bin/rocm-smi"
: > "$tmp/docker.log"
export PATH="$tmp/bin:$PATH"
export GPU_RUN_DIR=$tmp GPU_VLLM_UP="echo up >> $tmp/vllm.log" GPU_VLLM_DOWN="echo down >> $tmp/vllm.log"
export GPU_LLAMA_UP="echo up >> $tmp/vllm.log" GPU_LLAMA_DOWN="echo down >> $tmp/vllm.log"
export GPU_LLAMA_IS_UP="cat $tmp/serving 2>/dev/null"
export GPU_VLLM_IS_UP="$GPU_LLAMA_IS_UP"
export GPU_BUSY_CHECK=false GPU_PREEMPT_WAIT=1 GPU_SERVE_GRACE=3
export GPU_POLL_INTERVAL=0.1 GPU_VLLM_DIR="$tmp"
export GPU_WAKE_VENV="$tmp/venv"
export GPU_LLAMA_UP="$GPU_LLAMA_UP; echo yes > $tmp/serving"
export GPU_VLLM_UP="$GPU_VLLM_UP; echo yes > $tmp/serving"
export GPU_LLAMA_DOWN="$GPU_LLAMA_DOWN; rm -f $tmp/serving"
export GPU_VLLM_DOWN="$GPU_VLLM_DOWN; rm -f $tmp/serving"
gpu() { "$here/gpu" "$@"; }
fail() { echo "FAIL: $*"; exit 1; }

gpu serve; [[ $(cat "$tmp/gpu.state") == serve && ! -e $tmp/locks/rocm.state ]] || fail "serve state"

gpu train wake sleep 3 & t=$!
sleep 1
[[ $(gpu status) == "state: train:wake"* ]] || fail "train state while running"
[[ ! -e $tmp/locks/rocm.state ]] || fail "legacy state must stay absent during training"
gpu serve || fail "serve must defer while training"
[[ ! -e $tmp/serving && $(gpu status) == "state: train:wake"* ]] || fail "deferred serving must stay down"
"$here/gpu" run lfm sh -c 'echo ran > "$GPU_RUN_DIR/second"' & second=$!
sleep 0.2
[[ ! -e $tmp/second ]] || fail "second run must wait without overlap"
wait $t
wait "$second"
[[ -e $tmp/second ]] || fail "second run must eventually execute"
[[ $(gpu status) == "state: idle"* ]] || fail "idle after train"

echo "train:wake 999999" > "$tmp/gpu.state"
gpu serve || fail "dead train pid must count as idle"

gpu train wake --serve-after false 2>/dev/null && fail "failing cmd must fail"
for _ in {1..100}; do
  [[ $(cat "$tmp/gpu.state") == serve ]] && break
  sleep 0.1
done
[[ $(cat "$tmp/gpu.state") == serve ]] || fail "--serve-after must restart vLLM even when the run fails"

grep -q down "$tmp/vllm.log" || fail "train must stop vLLM"
"$here/gpu" train wake sleep 30 & t=$!
for _ in {1..100}; do
  read -r state owner worker _ < "$tmp/gpu.state"
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

"$here/gpu" train wake sleep 30 & t=$!
for _ in {1..100}; do
  read -r state owner worker _ < "$tmp/gpu.state"
  [[ $state == train:wake && -n ${worker:-} ]] && break
  sleep 0.05
done
[[ ${owner:-} == "$t" && -n ${worker:-} ]] || fail "hard-kill worker reservation"
kill -KILL "$t"; wait "$t" 2>/dev/null || true
gpu serve || fail "orphan worker must defer serving"
[[ ! -e $tmp/serving ]] || fail "orphan worker must block serving"
[[ $(gpu status) == *"owner alive: no"*"worker alive: yes"* ]] || fail "orphan status"
kill -TERM -- "-$worker"
for _ in {1..100}; do
  [[ $(gpu status) == "state: idle"* ]] && break
  sleep 0.05
done
[[ $(gpu status) == "state: idle"* ]] || fail "idle after orphan exits"

"$here/gpu" train wake sh -c 'sleep 30 & echo $! > "$1"; wait' sh "$tmp/descendant" & t=$!
for _ in {1..100}; do
  read -r state owner worker _ < "$tmp/gpu.state"
  [[ $state == train:wake && -n ${worker:-} && -s $tmp/descendant ]] && break
  sleep 0.05
done
[[ ${owner:-} == "$t" && -n ${worker:-} && -s $tmp/descendant ]] || fail "descendant reservation"
kill -KILL "$t"; wait "$t" 2>/dev/null || true
kill -KILL "$worker"
sleep 0.1
kill -0 "$(cat "$tmp/descendant")" || fail "regression requires a live descendant"
gpu serve || fail "live descendant must defer serving"
[[ ! -e $tmp/serving ]] || fail "live descendant must block serving"
"$here/gpu" run wake sh -c 'touch "$GPU_RUN_DIR/descendant-next"' & pending=$!
sleep 0.2
[[ ! -e $tmp/descendant-next ]] || fail "live descendant must block training"
kill -TERM "$pending"; wait "$pending" 2>/dev/null || true
[[ $(gpu status) == *"worker group alive: yes"* ]] || fail "descendant status"
kill -TERM -- "-$worker"
for _ in {1..100}; do
  [[ $(gpu status) == "state: idle"* ]] && break
  sleep 0.05
done
[[ $(gpu status) == "state: idle"* ]] || fail "idle only after descendants stop"

gpu stop
exec 7>"$tmp/gpu.watch.lock"
flock 7
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
exec 7>&-
gpu serve vllm
before=$(grep -c '^up$' "$tmp/vllm.log")
"$here/gpu" run host -- sh -ec 'mkdir "$GPU_RUN_DIR/active"; echo first >> "$GPU_RUN_DIR/order"; touch "$GPU_RUN_DIR/ready"; while [ ! -e "$GPU_RUN_DIR/go" ]; do sleep .1; done; rmdir "$GPU_RUN_DIR/active"' & first=$!
for _ in {1..100}; do [[ -e $tmp/ready ]] && break; sleep .05; done
[[ -e $tmp/ready && ! -e $tmp/serving ]] || fail "turn must stop inference"
"$here/gpu" run wake sh -ec 'mkdir "$GPU_RUN_DIR/active"; echo second >> "$GPU_RUN_DIR/order"; rmdir "$GPU_RUN_DIR/active"' & second=$!
for _ in {1..100}; do [[ $(gpu status) == *"wake pid=$second"* ]] && break; sleep .05; done
[[ $(gpu status) == *"wake pid=$second"* ]] || fail "second ticket registered"
"$here/gpu" run host sh -ec 'mkdir "$GPU_RUN_DIR/active"; echo third >> "$GPU_RUN_DIR/order"; rmdir "$GPU_RUN_DIR/active"' & third=$!
for _ in {1..100}; do [[ $(gpu status) == *"host pid=$third"* ]] && break; sleep .05; done
[[ $(gpu status) == *"host pid=$third"* ]] || fail "third ticket registered"
"$here/gpu" run doomed touch "$tmp/doomed-ran" & doomed=$!
for _ in {1..100}; do [[ $(gpu status) == *"doomed pid=$doomed"* ]] && break; sleep .05; done
[[ $(gpu status) == *"doomed pid=$doomed"* ]] || fail "doomed ticket registered"
kill -KILL "$doomed"; wait "$doomed" 2>/dev/null || true
touch "$tmp/go"
wait "$first"; wait "$second"; wait "$third"
[[ $(cat "$tmp/order") == $'first\nsecond\nthird' ]] || fail "FIFO order without overlap"
[[ $(grep -c '^up$' "$tmp/vllm.log") == "$before" && ! -e $tmp/serving ]] || fail "no serving between turns or before grace"
[[ ! -e $tmp/doomed-ran ]] || fail "killed waiter must not execute"
sleep 1
[[ ! -e $tmp/serving ]] || fail "serving must wait for the grace period"
for _ in {1..100}; do [[ -e $tmp/serving ]] && break; sleep .1; done
[[ -e $tmp/serving && $(grep -c '^up$' "$tmp/vllm.log") -eq $((before + 1)) ]] || fail "detached serving resumes once after grace"
[[ -z $(ls -A "$tmp/gpu.queue") ]] || fail "dead ticket pruned"
[[ $(gpu status) == *"default runtime: vllm"*"default runtime up: yes"* ]] || fail "persisted default and actual service status"

GPU_BUSY_CHECK='touch "$GPU_RUN_DIR/unexpected-busy-check"; true' \
  "$here/gpu" run host true
[[ ! -e $tmp/unexpected-busy-check && ! -e $tmp/serving ]] || fail "vllm must skip llama preemption checks"

gpu serve llama
GPU_PREEMPT_WAIT=5 GPU_BUSY_CHECK='touch "$GPU_RUN_DIR/busy-checked"; test ! -e "$GPU_RUN_DIR/slots-idle"' \
  "$here/gpu" run host touch "$tmp/after-busy" & busy=$!
for _ in {1..100}; do [[ -e $tmp/busy-checked ]] && break; sleep .05; done
[[ -e $tmp/busy-checked && -e $tmp/serving && ! -e $tmp/after-busy ]] || fail "busy slots delay preemption"
touch "$tmp/slots-idle"
wait "$busy"
[[ -e $tmp/after-busy && ! -e $tmp/serving ]] || fail "idle slots allow the turn"

gpu serve llama
port=$(python3 - <<'PORT'
import socket
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PORT
)
before=$SECONDS
env -u GPU_BUSY_CHECK GPU_LLAMA_SLOTS_URL="http://127.0.0.1:$port/slots" GPU_PREEMPT_WAIT=10 \
  "$here/gpu" run host touch "$tmp/unreachable-slots"
[[ -e $tmp/unreachable-slots && ! -e $tmp/serving ]] || fail "unreachable slots allow the turn"
(( SECONDS - before < 5 )) || fail "unreachable slots must not wait out preemption"

gpu stop
sleep 30 & legacy=$!
echo "train:wake $legacy" > "$tmp/gpu.state"
"$here/gpu" run host touch "$tmp/legacy-next" & pending=$!
sleep .3
gpu serve
[[ ! -e $tmp/serving && ! -e $tmp/legacy-next ]] || fail "legacy owner blocks serving and turns"
(
  flock 9
  echo idle > "$tmp/gpu.state"
) 9>"$tmp/gpu.lock"
kill "$legacy"; wait "$legacy" 2>/dev/null || true
gpu serve
wait "$pending"
[[ -e $tmp/legacy-next ]] || fail "legacy exit releases queued turn"
gpu stop
[[ $(gpu status) == *"default runtime: none"* ]] || fail "stop disables automatic serving"
sleep 4
[[ ! -e $tmp/serving ]] || fail "stop remains stopped after grace"
echo "gpu self-check: ok"
