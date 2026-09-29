"""
slow_or_dead.py -- is the server slow, or is it dead?

What it shows: the monitor (rank 0) sends one request to the server (rank 1)
and waits TIMEOUT seconds. The server is either slow (replies after 3 s) or
dead (never replies). The monitor prints the SAME thing in both cases: from
the outside, slow and dead look exactly alike.

Run:  mpiexec -n 2 python slow_or_dead.py slow
      mpiexec -n 2 python slow_or_dead.py dead
      mpiexec -n 2 python slow_or_dead.py slow 5     (timeout = 5 s)
"""
import sys
import time

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mode = sys.argv[1] if len(sys.argv) > 1 else "slow"         # "slow" or "dead"
timeout = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0  # seconds

if comm.Get_size() != 2:
    if rank == 0:
        print("Run with exactly 2 processes: mpiexec -n 2 python slow_or_dead.py slow")
    sys.exit(0)

if rank == 1:  # SERVER
    comm.recv(source=0)
    if mode == "slow":
        time.sleep(3)
        comm.send("pong", dest=0)
    # If dead, we simply never reply. (A real crash would make MPI kill the
    # whole job -- MPI assumes processes never fail -- so we fake it.)

if rank == 0:  # MONITOR
    comm.send("ping", dest=1)
    start = time.time()
    # iprobe asks "has a message arrived?" without blocking, so we can give
    # up after the timeout. A plain recv would wait forever if the server is dead.
    while time.time() - start < timeout:
        if comm.iprobe(source=1):
            comm.recv(source=1)
            print(f"monitor: reply after {time.time() - start:.1f} s -> server is ALIVE")
            break
        time.sleep(0.01)
    else:
        print(f"monitor: no reply within {timeout} s -> server is DEAD")

comm.Barrier()  # wait until both are done, then reveal the truth
if rank == 1:
    print(f"truth:   the server was {'just slow (3 s)' if mode == 'slow' else 'really dead'}")
