"""
hello_mpi.py -- a first look at MPI.

What it shows: several separate processes running the same program, each
finding out who it is (its rank), and then two of them exchanging messages.

Run:  mpiexec -n 4 python hello_mpi.py
"""
from mpi4py import MPI

# ---------------------------------------------------------------------------
# Part 1 -- "Who am I?"
# ---------------------------------------------------------------------------
# `mpiexec -n 4` starts 4 PROCESSES. Each one runs this same file from the
# top, with its own separate memory. Nothing is shared between them.
#
# A COMMUNICATOR is a group of processes that can talk to each other.
# MPI.COMM_WORLD is the default one: it contains every process mpiexec
# started.
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # RANK: my id in the group, 0, 1, ..., size-1
size = comm.Get_size()  # SIZE: how many processes are in the group
host = MPI.Get_processor_name()  # the machine this process runs on

# The only thing that makes the processes behave differently is their rank.
# Run this several times: the order of these lines changes from run to run.
# No process knows what the others are doing "right now" -- there is no
# global order of events.
print(f"Hello from rank {rank} of {size} on {host}")

# A barrier makes every process wait here until all have arrived. 
# It does NOT guarantee the print order above.
comm.Barrier()

# ---------------------------------------------------------------------------
# Part 2 -- "Talking to each other"
# ---------------------------------------------------------------------------
# There is no shared memory, so the only way to share information is to
# send a POINT-TO-POINT MESSAGE from one process to another.
if size < 2:
    if rank == 0:
        print("Part 2 needs at least 2 processes. Try: mpiexec -n 2 python hello_mpi.py")
elif rank == 0:
    # send(obj, dest, tag): any Python object can be sent. The TAG is a
    # number we attach so the receiver can tell kinds of messages apart.
    comm.send({"greeting": "hi", "from": 0}, dest=1, tag=0)
    # recv BLOCKS: rank 0 stops here until a matching message arrives.
    # SOURCE and TAG say which message we want: from rank 1, with tag 1.
    reply = comm.recv(source=1, tag=1)
    print(f"rank 0 got a reply: {reply}")
elif rank == 1:
    msg = comm.recv(source=0, tag=0)  # waits for rank 0's message
    print(f"rank 1 received:    {msg}")
    comm.send({"greeting": "hello back", "from": 1}, dest=0, tag=1)
# Ranks 2, 3, ... do nothing in Part 2; they simply reach the end and exit.
