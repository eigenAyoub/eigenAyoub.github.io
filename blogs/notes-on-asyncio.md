---

---


Lot of cool things written with `asyncio`:

* sglang
* vllm?
* check some of aleph alpha shit.



### await core(),  await task

* `await task` and `await coro()` both pause the current coroutine until the task finishes or raises.

* When you create a task, you automatically schedule it to run.  
	* Technically you don't to `await` it, but do it regardless to enforce order and catch exceptions.

### `asyncio.Queue` vs `queue.Queue`

* `asyncio.Queue` is non-blocking for the thread; 
* `await q.get()` suspends only the coroutine, and yields control to the loop.
* `queue.Queue.get()` blocks the thread.


### `await q.put(x)`:

* Natural backpressure, if the queue is full, it just waits until space is free. 

### example:

```python
import asyncio

q = asyncio.Queue(maxsize=1)

async def prod():
    await q.put(0)
    print("P: waiting for space")
	await q.put(1)
	print("P: resumed")

async def cons():
    await asyncio.sleep(0.1)
	_ = await q.get()

async def main():
	await asyncio.gather(prod(), cons())

asyncio.run(main())
```

### graceful cancellation

What's the proper way of cancelling a task:


```python
import asyncio

async def tight_loop_no_yield():
    try:
        for _ in range(50_000_000):
            pass  # no await -> not cancellable until loop ends
        print("finished without yielding")
    except asyncio.CancelledError:
        print("cleanup (never reached here)")
        raise

async def tight_loop_with_yield():
    try:
        for i in range(50_000_000):
            if i % 1_000_000 == 0:
                await asyncio.sleep(0)  # cancellation point
    except asyncio.CancelledError:
        print("cleanup (with yield)")
        raise

async def main():
    # Case 1: no yield -> cancel request is delayed
    t1 = asyncio.create_task(tight_loop_no_yield())
    await asyncio.sleep(0.05)
    t1.cancel()
    try:
        await t1
    except asyncio.CancelledError:
        print("caller saw cancellation (likely only if loop finished first)")

    # Case 2: yield -> cancels quickly
    t2 = asyncio.create_task(tight_loop_with_yield())
    await asyncio.sleep(0.05)
    t2.cancel()
    try:
        await t2
    except asyncio.CancelledError:
        print("caller saw cancellation promptly")

asyncio.run(main())
```





### some funsies with cancellation:
 

```python
import asyncio

# A) catch + RE-RAISE  -> caller sees cancellation
async def w_reraise():
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        print("cleanup A")
        raise  # <-- re-raise same CancelledError

# B) catch + SWALLOW  -> caller thinks success (BAD unless intentional)
async def w_swallow():
    try:
        await asyncio.sleep(10)
    except asyncio.CancelledError:
        print("cleanup B (swallowed)")
        return "ok"  # <-- cancellation hidden

# C) NO except -> finally still runs; cancellation propagates
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
        r = await task              # joins; if cancelled properly, raises here
        print("result:", r)
    except asyncio.CancelledError:
        print("caller: saw cancellation")

asyncio.run(demo(w_reraise))  # raises -> caller sees cancellation
asyncio.run(demo(w_swallow))  # prints "result: ok" (cancellation hidden)
asyncio.run(demo(w_noexcept)) # raises -> caller sees cancellation
```






