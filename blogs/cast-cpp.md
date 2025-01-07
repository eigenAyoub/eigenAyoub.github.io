---
layout: default
title:
permalink: /blogs/cast-cpp/
---

## Casting Operators: `static_cast` `reinterpret_cast`.

Both these two operators deal with `void *` pointers.

** `void*` - The Universal Pointer**

A `void*` pointer can hold the address of any data type. However, you can't directly dereference it because the compiler doesn't know the underlying type it points to. To use the data, you must cast it back to the appropriate type.

**`static_cast` - The Safe Bet (Usually)**

`static_cast` is generally considered the safer option when dealing with related types. It performs compile-time checks and ensures that the conversion is at least somewhat plausible.

**When `static_cast` works with `void*`:**

1. **Casting to `void*`:** You can safely cast any pointer type to `void*` using `static_cast`. This is an implicit conversion, so `static_cast` is not strictly required, but using it can make it clear what you are doing.

    ```cpp
    int* ptr = new int(10);
    void* vptr = static_cast<void*>(ptr); // Upcast to void*
    ```

2. **Casting from `void*` back to the *original* type:**  If you know the original type a `void*` was cast from, you can safely use `static_cast` to convert it back.

    ```cpp
    int* originalPtr = static_cast<int*>(vptr); // Downcast back to int*
    ```

**When `static_cast` might fail:**

`static_cast` will not let you directly cast a `void*` to an unrelated pointer type. If you made a mistake and `vptr` doesn't actually hold an `int*`, but `static_cast` it back to an `int*`, you will get undefined behavior.

**`reinterpret_cast` - The Power (and Peril) of Reinterpretation**

`reinterpret_cast` is the most powerful and potentially dangerous casting operator. It essentially tells the compiler, "Treat this memory address as if it holds data of this other type," without any real checks.

**`reinterpret_cast` and `void*`:**

1. **Casting between `void*` and any pointer type:** You can use `reinterpret_cast` to cast between `void*` and any other pointer type, even unrelated ones.

    ```cpp
    char* charPtr = reinterpret_cast<char*>(vptr); // Reinterpreting int* as char*
    ```

**The Dangers of `reinterpret_cast`:**

*   **Undefined Behavior:** If you `reinterpret_cast` a `void*` to an incorrect type, your program will likely exhibit undefined behavior, leading to crashes or unpredictable results. This is because `reinterpret_cast` does not check if the cast actually makes sense, it just forces it to happen.
*   **Portability Issues:** `reinterpret_cast` can lead to code that is less portable, as the results might depend on the specific memory layout and architecture.

**Key Takeaways:**

*   Prefer `static_cast` when casting to `void*` and back to the *original* type. It provides some level of safety.
*   Use `reinterpret_cast` with extreme caution when dealing with `void*`. Only use it when you are absolutely sure about the underlying type and the implications of reinterpreting the memory.
*   Always double-check your logic when using `reinterpret_cast`. One wrong move can lead to hard-to-debug issues.

**In essence, `static_cast` is like a guided conversion, while `reinterpret_cast` is like a forceful override. Choose wisely!**


An interesting point:


