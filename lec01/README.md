# Lecture 1: hands-on MPI examples

## 1. What is in this folder

- `hello_mpi.py` -- MPI "hello world": processes, ranks, and a first exchange of messages.
- `slow_or_dead.py` -- a timeout-based failure detector that cannot tell a slow server from a crashed one.

## 2. Setup

You need an MPI implementation (we use Open MPI) and the Python package `mpi4py`.
Pick one option:

- **macOS**
  ```
  brew install open-mpi
  pip install mpi4py
  ```
- **Ubuntu/Debian**
  ```
  sudo apt install openmpi-bin libopenmpi-dev python3-mpi4py python-is-python3
  ```
- **Any OS with conda**
  ```
  conda install -c conda-forge mpi4py openmpi
  ```

Check that it works (it should print `0` and `1`, in some order):

```
mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
```

Notes:
- If you get `unable to find the specified executable file ... python`, your system only
  has `python3`: type `python3` instead of `python` in every command.
- On some systems the launcher is called `mpirun` instead of `mpiexec`. They take the same options.
- If `-n` is larger than the number of cores on your laptop, Open MPI refuses to start.
  Add `--oversubscribe`, e.g. `mpiexec --oversubscribe -n 4 python hello_mpi.py`.

## 3. Running `hello_mpi.py`

```
mpiexec -n 4 python hello_mpi.py
```

Run it several times and look at the order of the "Hello" lines. Does it stay the same?
Then try `-n 1` and `-n 2`.

## 4. Running `slow_or_dead.py`

```
mpiexec -n 2 python slow_or_dead.py
```

Rank 0 is the monitor and rank 1 is the server. Each round, the monitor sends a request
and waits `--timeout` seconds for a reply, then declares the server ALIVE or DEAD. Each line
also shows what really happened, so you can see when the monitor was wrong. Run `python slow_or_dead.py --help` to see all options.

### Try different timeouts

```
mpiexec -n 2 python slow_or_dead.py --timeout 0.1
mpiexec -n 2 python slow_or_dead.py --timeout 0.5
mpiexec -n 2 python slow_or_dead.py --timeout 1.0
mpiexec -n 2 python slow_or_dead.py --timeout 2.5
```

With the default seed you get:

| timeout | slow server wrongly declared DEAD | real crash noticed after |
|---|---|---|
| 0.1 s | 9 times | 0.1 s |
| 0.5 s | 4 times | 0.5 s |
| 1.0 s | 3 times | 1.0 s |
| 2.5 s | 0 times | 2.5 s |

Short timeout: fast detection, many mistakes. Long timeout: no mistakes, slow detection.
Zero mistakes is only possible here because we know no reply takes longer than 2.0 s.
A real network gives no such bound, so no timeout is always right.

## 5. Glossary

| Term | Meaning |
|---|---|
| process | A running program with its own memory. MPI processes share no memory. |
| `mpiexec` | The launcher. `mpiexec -n 4 python f.py` starts 4 processes, all running `f.py`. |
| communicator | A group of processes that can send messages to each other. |
| `COMM_WORLD` | The communicator that contains every process started by `mpiexec`. |
| rank | A process's id inside a communicator: 0, 1, ..., size-1. |
| size | The number of processes in a communicator. |
| `send` | Send a Python object to another process (`comm.send(obj, dest=1, tag=0)`). |
| `recv` | Receive a message (`comm.recv(source=0, tag=0)`). It waits (blocks) until a matching message arrives. |
| tag | A number attached to a message so the receiver can pick which message it wants. |
| `iprobe` | Check whether a matching message is waiting, without receiving it and without waiting. |
