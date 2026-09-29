"""
slow_or_dead.py -- is the server slow, or is it dead?

What it shows: a MONITOR (rank 0) sends requests to a SERVER (rank 1) and
waits a fixed TIMEOUT for each reply. The server is sometimes slow, and at
some point it crashes. The monitor cannot tell "slow" from "dead" -- it only
sees "no reply yet". A short timeout wrongly suspects slow rounds (FALSE
SUSPICIONS); a long timeout notices the real crash late.
Lecture link: unbounded message delay; failure detection by timeout.

Run:  mpiexec -n 2 python slow_or_dead.py
      mpiexec -n 2 python slow_or_dead.py --timeout 0.1 --seed 42
"""
import argparse
import random
import time

from mpi4py import MPI

STOP_TAG = 10000    # monitor -> server: "the experiment is over"
ORACLE_TAG = 10001  # server -> monitor: the ground truth (for scoring only)
GRACE = 2.5         # seconds to wait for late replies; > the max slow delay

parser = argparse.ArgumentParser()
parser.add_argument("--rounds", type=int, default=20, help="number of request rounds")
parser.add_argument("--timeout", type=float, default=0.5, help="seconds to wait per round")
parser.add_argument("--slow-prob", type=float, default=0.2, help="chance a reply is slow")
parser.add_argument("--crash-round", type=int, default=-1,
                    help="round at which the server crashes; -1 = random in 2nd half, 0 = never")
parser.add_argument("--seed", type=int, default=42, help="random seed (reproducible demos)")
args = parser.parse_args()  # both ranks run this line and get the same args

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def server():
    """Rank 1: answer each request after a random delay, until it 'crashes'."""
    rng = random.Random(args.seed)
    crash_round = args.crash_round
    if crash_round == -1:
        crash_round = rng.randint(args.rounds // 2 + 1, args.rounds)
    crashed = False
    truth = {}    # round -> ("normal" | "slow", delay) or ("crashed", None)
    pending = []  # replies not sent yet: (time to send, round)
    st = MPI.Status()

    # Each request gets its own delay, as if the server had one worker per
    # request: a slow reply does not hold up the replies behind it.
    while True:
        now = time.monotonic()
        for due, r in sorted(pending):
            if due <= now:
                comm.send("pong", dest=0, tag=r)  # the reply's tag is its round
                pending.remove((due, r))

        if not comm.iprobe(source=0, tag=MPI.ANY_TAG, status=st):
            time.sleep(0.001)
            continue
        r = st.Get_tag()
        comm.recv(source=0, tag=r)
        if r == STOP_TAG:
            for due, k in sorted(pending):  # send any reply still owed
                time.sleep(max(0.0, due - time.monotonic()))
                comm.send("pong", dest=0, tag=k)
            break
        if r == crash_round:
            # A simulated crash: from now on we never reply (replies already
            # on their way still arrive). The process keeps running only so
            # the program can end cleanly. If an MPI process REALLY died, MPI
            # would kill the whole job -- MPI assumes processes do not fail.
            crashed = True
        if crashed:
            truth[r] = ("crashed", None)
            continue
        if rng.random() < args.slow_prob:
            kind, delay = "slow", rng.uniform(0.6, 2.0)
        else:
            kind, delay = "normal", rng.uniform(0.05, 0.20)
        truth[r] = (kind, delay)
        pending.append((time.monotonic() + delay, r))  # reply `delay` s from now

    # Cheating on purpose: a real monitor never learns the truth. We reveal
    # it only so that we can score the monitor's verdicts.
    comm.send({"truth": truth, "crash_round": crash_round if crashed else None},
              dest=0, tag=ORACLE_TAG)


def monitor():
    """Rank 0: send one request per round and judge ALIVE or SUSPECTED."""
    sent_at = {}       # round -> time the request was sent
    verdict = {}       # round -> "ALIVE" or "SUSPECTED"
    suspected_at = {}  # round -> time the monitor gave up on it
    late = {}          # round -> seconds after sending that the late reply came
    st = MPI.Status()  # iprobe fills this in with the tag of the waiting message

    def drain_late_replies(before_round):
        # iprobe = "is there a message waiting?" It never blocks. Every reply
        # carries its round number as its tag, so an old reply can never be
        # mistaken for the reply of the current round.
        while comm.iprobe(source=1, tag=MPI.ANY_TAG, status=st) and st.Get_tag() < before_round:
            k = st.Get_tag()
            comm.recv(source=1, tag=k)
            late[k] = time.monotonic() - sent_at[k]
            print(f"   late reply for round {k} arrived -- round {k} was a FALSE SUSPICION")

    print(f"monitor: {args.rounds} rounds, timeout {args.timeout:.3f} s\n")
    for r in range(1, args.rounds + 1):
        drain_late_replies(r)
        comm.send("ping", dest=1, tag=r)
        sent_at[r] = time.monotonic()

        # Poll until the reply for round r is waiting, or the timeout expires.
        # We use iprobe + recv instead of irecv + cancel: we only recv a
        # message once we know it is there, so no half-finished (cancelled)
        # receive requests are left lying around.
        arrived = False
        while time.monotonic() - sent_at[r] < args.timeout:
            if comm.iprobe(source=1, tag=r):
                arrived = True
                break
            time.sleep(0.001)

        if arrived:
            comm.recv(source=1, tag=r)
            verdict[r] = "ALIVE"
            print(f"round {r:2d} | ALIVE     | reply after {time.monotonic() - sent_at[r]:.3f} s")
        else:
            verdict[r] = "SUSPECTED"
            suspected_at[r] = time.monotonic()
            print(f"round {r:2d} | SUSPECTED | no reply within {args.timeout:.3f} s")

    # Give slow replies a chance to show up, so each one is accounted for.
    end = time.monotonic() + GRACE
    while time.monotonic() < end:
        drain_late_replies(args.rounds + 1)
        time.sleep(0.001)

    comm.send(None, dest=1, tag=STOP_TAG)
    # Messages between two processes arrive in the order they were sent, so
    # any reply still on its way comes before the oracle. Handle it as late.
    while True:
        msg = comm.recv(source=1, tag=MPI.ANY_TAG, status=st)
        if st.Get_tag() == ORACLE_TAG:
            truth, crash_round = msg["truth"], msg["crash_round"]
            break
        k = st.Get_tag()
        late[k] = time.monotonic() - sent_at[k]
        print(f"   late reply for round {k} arrived -- round {k} was a FALSE SUSPICION")

    # ------------------------- score the detector -------------------------
    print("\nround | verdict   | truth             | result")
    print("------+-----------+-------------------+------------------")
    n_alive = n_false = n_correct = 0
    for r in range(1, args.rounds + 1):
        kind, delay = truth[r]
        shown = kind if delay is None else f"{kind} ({delay:.3f} s)"
        if kind == "crashed":
            result = "correct suspicion"
            n_correct += 1
        elif verdict[r] == "ALIVE":
            result = "ok"
            n_alive += 1
        else:
            result = "FALSE SUSPICION"
            n_false += 1
        print(f"{r:5d} | {verdict[r]:9s} | {shown:17s} | {result}")

    if crash_round is None:
        detection = "n/a (no crash)"
    else:
        detection = f"{suspected_at[crash_round] - sent_at[crash_round]:.3f} s (crash in round {crash_round})"
    print(f"\ncorrect ALIVE verdicts : {n_alive}")
    print(f"FALSE SUSPICIONS       : {n_false}  (suspected, but the server was only slow)")
    print(f"correct suspicions     : {n_correct}  (the server really had crashed)")
    print(f"crash detection delay  : {detection}")
    print("\nA shorter timeout detects crashes faster but wrongly suspects slow nodes.")
    print("A longer timeout avoids false suspicions but reacts slowly to real crashes.")
    print("No timeout is always right in an asynchronous system.")


if size != 2:
    if rank == 0:
        print("This demo needs exactly 2 processes. Try: mpiexec -n 2 python slow_or_dead.py")
elif rank == 0:
    monitor()
else:
    server()
