"""
slow_or_dead.py -- is the server slow, or is it dead?

What it shows: a MONITOR (rank 0) sends a request to a SERVER (rank 1) each
round and waits TIMEOUT seconds for the reply. No reply in time -> the
monitor suspects the server is dead. But the server is sometimes just slow,
and at some point it really crashes. A short timeout wrongly suspects slow
rounds (FALSE SUSPICIONS); a long timeout notices the real crash late.

Run:  mpiexec -n 2 python slow_or_dead.py
      mpiexec -n 2 python slow_or_dead.py --timeout 0.1
"""
import argparse
import random
import time

from mpi4py import MPI

# After guessing, the monitor keeps listening this long, only so that we can
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
    """Rank 0: each round, send a request and guess ALIVE or SUSPECTED."""
    n_ok = n_false = n_correct = 0
    print(f"timeout = {args.timeout:.3f} s\n")
    print("round | guess     | what really happened         | result")
    print("------+-----------+------------------------------+------------------")
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

        # The guess uses only what the monitor saw within the timeout.
        if delay is not None and delay <= args.timeout:
            guess, truth, result = "ALIVE", f"reply after {delay:.3f} s", "ok"
            n_ok += 1
        elif delay is not None:
            guess, truth, result = "SUSPECTED", f"slow: reply after {delay:.3f} s", "FALSE SUSPICION"
            n_false += 1
        else:
            guess, truth, result = "SUSPECTED", "crashed: no reply", "correct suspicion"
            n_correct += 1
        print(f"{r:5d} | {guess:9s} | {truth:28s} | {result}")

    detection = f"{args.timeout:.3f} s (= the timeout)" if n_correct else "n/a (no crash)"
    print(f"\ncorrect ALIVE guesses : {n_ok}")
    print(f"FALSE SUSPICIONS      : {n_false}  (suspected, but the server was only slow)")
    print(f"correct suspicions    : {n_correct}  (the server really had crashed)")
    print(f"crash detection delay : {detection}")
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
