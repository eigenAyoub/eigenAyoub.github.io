---
layout: default 
title: 
permalink: /blogs/algo/boyer-moore
---

The [Boyer-Moore majority vote algorithm](https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore_majority_vote_algorithm)
is relevant when you have a sequence of elements, and you want the majority element, **without** storing a count for each distinct element.
So you want to determine the majority element(s) in linear time and constant space. Examples:

- **[169. Majority Element](https://leetcode.com/problems/majority-element/):** Find the one element appearing more than `n / 2` times. 

- **[229. Majority Element II](https://leetcode.com/problems/majority-element-ii/):** Find all elements appearing more than `n / 3` times.

Both are obvious to solve using a `Counter`, but that's not interesting, we want **(1) extra space**.
    
For problem 169, the algorithm stores only one candidate and one counter and requires only one pass through the sequence, since the majority element is guaranteed to exist. Here is a pseudocode: 


```text
m ← undefined
c ← 0

for each element x in the input sequence:
    if c = 0:
        m ← x
        c ← 1
    else if m = x:
        c ← c + 1
    else:
        c ← c - 1

return m
```


For problem 229, we track two candidates and two counters, then verify both candidates in a second pass. 

**Why a second pass:** The algorithm always outputs something, so another verification pass has to be done to check whether the final candidate passes the majority threshold. Here is a solution for problem 229:


```python
def majorityElement(nums: list[int]) -> list[int]:
    c1 = c2 = None
    f1 = f2 = 0

    # Find potential majority elements
    for num in nums:
        if num == c1:
            f1 += 1
        elif num == c2:
            f2 += 1
        elif f1 == 0:
            c1, f1 = num, 1
        elif f2 == 0:
            c2, f2 = num, 1
        else:
            f1 -= 1
            f2 -= 1

    # verify
    count1 = count2 = 0
    for num in nums:
        if num == c1:
            count1 += 1
        elif num == c2:
            count2 += 1

    threshold = len(nums) // 3
    result = []
    if count1 > threshold:
        result.append(c1)
    if count2 > threshold:
        result.append(c2)

    return result
```