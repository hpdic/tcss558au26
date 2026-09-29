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
and waits `--timeout` seconds for a reply. At the end, the program reveals what really
happened and scores the monitor's guesses. Run `python slow_or_dead.py --help` to see all options.

### In-class exercise

Run the program with `--timeout 0.1`, `0.5`, `1.0`, and `2.5`, keeping the same `--seed`:

```
mpiexec -n 2 python slow_or_dead.py --timeout 0.1 --seed 42
```

For each timeout, record the number of FALSE SUSPICIONS and the crash detection delay.

- **Question 1:** Which timeout would you choose, and why?
- **Question 2:** Could *any* timeout give zero false suspicions *and* fast detection?
  What would you need to know about the network to make that possible?
- **Question 3:** What changes if the server's delays come from a distribution you do
  not know in advance?

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
