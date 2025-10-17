---
layout: default
title:
permalink: /blogs/asyncio/
---

**Some scattered notes on `asyncio`:**

gpt, thanks for all the discussions, code snippets, and for polishing this.

* Main references: 
  * [https://docs.python.org/3/library/asyncio-task.html](https://docs.python.org/3/library/asyncio-task.html)
  * [https://realpython.com/async-io-python/](https://realpython.com/async-io-python/?utm_source=chatgpt.com)

## Table of contents

- [Task scheduling & awaiting](#task-scheduling-awaiting)
- [`asyncio.Queue` vs `queue.Queue`](#asyncio-queue-vs-queue-queue)
- [Why `await q.put(x)`](#why-await-q-put)
- [Graceful cancellation](#graceful-cancellation)
  - [CPU loops need explicit cancellation points](#cpu-cancellation-points)
- [Cancellation behaviors compared](#cancellation-behaviors-compared)
- [`asyncio.gather` (ordering & failures)](#asyncio-gather)
- [`asyncio.TaskGroup`](#taskgroup)


## Difference between `asyncio.wait()` and `asyncio.wait_for()`:

`wait()` doesn't cancel the futures when a timeout occurs, `wait_for()` does.





## Task scheduling & awaiting {#task-scheduling-awaiting}

- `await coro()` **pauses the current coroutine** until `coro()` completes or raises.
- `task = asyncio.create_task(coro())` **schedules** `coro()` to run immediately (next loop tick).
- You *should usually* `await task` to:
  - observe exceptions (avoid “Task exception was never retrieved”),
  - enforce ordering (join point),
  - coordinate shutdown (cancellation + cleanup).

```python
import asyncio

async def work():
    await asyncio.sleep(0.1)
    return 42

async def main():
    t = asyncio.create_task(work())  # starts promptly
    # ... do other work ...
    res = await t                    # join, get exceptions/result
    print(res)

asyncio.run(main())
```

## `asyncio.Queue` vs `queue.Queue` {#asyncio-queue-vs-queue-queue}

- `asyncio.Queue`: `await q.get()` suspends **only the coroutine** (thread keeps serving other tasks).
- `queue.Queue`: `q.get()` **blocks the OS thread** (unsuitable inside async code).

```python
# asyncio (non-blocking thread)
import asyncio
async def consumer(q): item = await q.get()

# threading (blocks thread)
import queue
q = queue.Queue()
def consumer_thread(): item = q.get()
```

## Why `await q.put(x)` {#why-await-q-put}

- Natural **backpressure**: if the queue is full (bounded `maxsize`), the producer **awaits** until a consumer frees a slot, without blocking the thread.

```python
import asyncio
q = asyncio.Queue(maxsize=1)

async def prod():
    await q.put(0)
    print("P: waiting for space")
    await q.put(1)            # waits here until a get() frees space
    print("P: resumed")

async def cons():
    await asyncio.sleep(0.1)
    _ = await q.get()

async def main():
    await asyncio.gather(prod(), cons())

asyncio.run(main())
```

## Graceful cancellation: {#graceful-cancellation}

**Rule:** cancel, then **await the task** so its cleanup runs and the exception is observed.

```python
import asyncio

async def worker():
    try:
        print("worker: start")
        await asyncio.sleep(10)          # cancellation lands here
        print("worker: done")            # not reached if cancelled
    except asyncio.CancelledError:
        print("worker: cleanup")
        raise                            # re-raise so caller sees cancellation

async def main():
    task = asyncio.create_task(worker())
    await asyncio.sleep(0.1)             # let it start
    task.cancel()                        # request cancellation
    try:
        await task                       # join -> runs cleanup, then raises here
    except asyncio.CancelledError:
        print("main: saw cancellation")

asyncio.run(main())
```

### CPU loops need explicit cancellation points {#cpu-cancellation-points}

Cancellation is **cooperative**. Add `await asyncio.sleep(0)` (or another await) in long loops.

*Cooperative* here means an asyncio task is not force-killed. The event loop **requests** cancellation and the coroutine **must reach a cancellation point** (an `await` that can suspend) to observe it.

```python
import asyncio

async def tight_loop_no_yield():
    for _ in range(50_000_000):
        pass  # no await -> cannot be cancelled until loop ends

async def tight_loop_with_yield():
    for i in range(50_000_000):
        if i % 1_000_000 == 0:
            await asyncio.sleep(0)  # cancellation point
```

## Cancellation behaviors compared {#cancellation-behaviors-compared}

```python
import asyncio

# A) Catch + re-raise  -> caller sees cancellation (recommended for custom cleanup)
async def w_reraise():
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        print("cleanup A")
        raise                      # keep cancellation visible

# B) Catch + swallow   -> caller thinks success (only if intentional)
async def w_swallow():
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        print("cleanup B (swallowed)")
        return "ok"                # hides cancellation

# C) No except         -> finally still runs; cancellation propagates
async def w_noexcept():
    try:
        await asyncio.sleep(10)
    finally:
        print("cleanup C (finally)")

async def demo(worker):
    task = asyncio.create_task(worker())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        print("awaiting task...")
        r = await task
        print("result:", r)
    except asyncio.CancelledError:
        print("caller: saw cancellation")

asyncio.run(demo(w_reraise))   # raises -> caller sees cancellation
asyncio.run(demo(w_swallow))   # prints "result: ok" (cancellation hidden)
asyncio.run(demo(w_noexcept))  # raises -> caller sees cancellation
```

## `asyncio.gather` (ordering & failures) {#asyncio-gather}

- Starts all awaitables concurrently; **results are returned in the same order as arguments**.
- On first exception it cancels the rest (default). Use `return_exceptions=True` to collect errors.

```python
import asyncio
async def slow(t): await asyncio.sleep(t); return t
async def main():
    res = await asyncio.gather(slow(0.2), slow(0.05), slow(0.1))
    print(res)  # [0.2, 0.05, 0.1] — argument order
asyncio.run(main())
```


## `asyncio.TaskGroup`: structured concurrency {#taskgroup}

**Why:** own a group of tasks with a clear lifetime. On any failure, siblings are **cancelled**, **awaited** for cleanup, and an `ExceptionGroup` is raised. Hence, no need for manual cancel/join loops, and prevents orphaned tasks.

**Use when:** supervising concurrent work you must finish/cancel **before** leaving a scope (e.g., multiple model backends, parallel pre/post-processing, multi-shard fetches).

```python
import asyncio

async def step(i):
    await asyncio.sleep(0.1 * i)
    if i == 2:
        raise RuntimeError("step 2 failed")
    return i

async def main():
    results = [None] * 4
    try:
        async with asyncio.TaskGroup() as tg:
            for i in range(4):
                t = tg.create_task(step(i))
                t.add_done_callback(lambda f, k=i: (
                    None if f.cancelled() or f.exception() else results.__setitem__(k, f.result())
                ))
        # Reached only if all tasks completed successfully
        print("all done:", results)
    except* RuntimeError as eg:
        print("TaskGroup caught:", eg)   # siblings already cancelled & cleaned

asyncio.run(main())
```

---

**Refs to explore:** sglang, vLLM, and Aleph Alpha repos use these patterns (queues, tasks, cancellation) in inference services.
