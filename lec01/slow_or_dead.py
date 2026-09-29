"""
slow_or_dead.py -- is the server slow, or is it dead?

What it shows: a MONITOR (rank 0) sends a request to a SERVER (rank 1) each
round and waits TIMEOUT seconds for the reply. No reply in time -> the
monitor declares the server DEAD. But the server is sometimes just slow, and
at some point it really crashes. A short timeout wrongly declares a slow
server dead; a long timeout notices the real crash late.

Run:  mpiexec -n 2 python slow_or_dead.py
      mpiexec -n 2 python slow_or_dead.py --timeout 0.1
"""
import argparse
import random
import time

from mpi4py import MPI

# After deciding, the monitor keeps listening this long, only so that we can
# see what really happened. It is longer than any slow reply (2.0 s), so "no
# reply by then" means the server really crashed. A real monitor cannot do
# this: in a real system there is no known upper bound on delay.
WAIT_FOR_TRUTH = 2.5

parser = argparse.ArgumentParser()
parser.add_argument("--rounds", type=int, default=12, help="number of request rounds")
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
    """Rank 1: reply to each request after a random delay, until it 'crashes'."""
    rng = random.Random(args.seed)
    crash_round = args.crash_round
    if crash_round == -1:
        crash_round = rng.randint(args.rounds // 2 + 1, args.rounds)
    for r in range(1, args.rounds + 1):
        comm.recv(source=0, tag=r)  # wait for the request of round r
        if crash_round and r >= crash_round:
            # A simulated crash: we just never reply. The process stays alive
            # only so the program ends cleanly. If an MPI process REALLY died,
            # MPI would kill the whole job -- MPI assumes processes never fail.
            continue
        if rng.random() < args.slow_prob:
            time.sleep(rng.uniform(0.6, 2.0))   # slow
        else:
            time.sleep(rng.uniform(0.05, 0.20))  # normal
        comm.send("pong", dest=0, tag=r)  # the tag says which round this answers

def monitor():
    """Rank 0: each round, send a request. No reply within the timeout -> DEAD."""
    n_wrong = 0
    crashed = False
    print(f"timeout = {args.timeout:.3f} s\n")
    print("round | monitor says | really")
    print("------+--------------+--------------------------------")
    for r in range(1, args.rounds + 1):
        comm.send("ping", dest=1, tag=r)
        start = time.monotonic()

        # iprobe asks "is a message from rank 1 with tag r waiting?" without
        # blocking, so we can give up after a while. A plain recv would wait
        # forever if the server has crashed.
        delay = None
        while time.monotonic() - start < WAIT_FOR_TRUTH:
            if comm.iprobe(source=1, tag=r):
                comm.recv(source=1, tag=r)
                delay = time.monotonic() - start
                break
            time.sleep(0.001)

        # The monitor decides using only the timeout.
        says = "ALIVE" if delay is not None and delay <= args.timeout else "DEAD"
        if delay is None:
            really = "dead"
            crashed = True
        elif says == "ALIVE":
            really = f"alive ({delay:.3f} s)"
        else:
            really = f"alive, just slow ({delay:.3f} s)  <-- WRONG"
            n_wrong += 1
        print(f"{r:5d} | {says:12s} | {really}")

    print(f"\nslow server wrongly declared DEAD : {n_wrong} times")
    if crashed:
        print(f"real crash noticed after         : {args.timeout:.3f} s (= the timeout)")
    print("\nA shorter timeout notices crashes faster but wrongly declares slow servers dead.")
    print("A longer timeout makes fewer mistakes but notices real crashes later.")
    print("No timeout is always right in an asynchronous system.")

if size != 2:
    if rank == 0:
        print("This demo needs exactly 2 processes. Try: mpiexec -n 2 python slow_or_dead.py")
elif rank == 0:
    monitor()
else:
    server()
