---
layout: default
title:
permalink: /blogs/rand-cpp/
---

## Demystifying Random Numbers and Uniform Distribution in C++

Random number generation is a cornerstone of many applications, from simulations and games to, importantly, machine learning. In this post, we'll dive into how C++ handles random numbers and explore how to generate numbers following a uniform distribution.

### The Foundation: `<random>`

Before C++11, generating truly random numbers wasn't straightforward. You'd often rely on `rand()` from `<cstdlib>`, which had limitations in terms of randomness quality and distribution control. C++11 revolutionized this with the `<random>` header.

`<random>` introduces two key concepts:

1. **Engines:** These are the heart of random number generation. They produce sequences of pseudo-random numbers based on an internal state. C++ provides various engines like `std::mt19937` (Mersenne Twister, generally preferred for its good statistical properties) and `std::linear_congruential_engine` (faster but potentially lower quality).

2. **Distributions:** Engines generate raw numbers. Distributions transform these raw numbers into numbers that follow a specific probability distribution (e.g., uniform, normal, Poisson).

### Generating Uniformly Distributed Random Numbers

The uniform distribution is one of the simplest yet most fundamental distributions. It means every number within a given range has an equal probability of being generated. Here's how to do it in C++:

```c++
#include <iostream>
#include <random>

int main() {
  // 1. Create a random number engine (Mersenne Twister in this case)
  std::random_device rd;  // Obtain a seed from the operating system
  std::mt19937 gen(rd()); // Seed the engine

  // 2. Define a uniform distribution
  //    - For integers between 1 and 10 (inclusive):
  std::uniform_int_distribution<> distrib_int(1, 10);
  //    - For real numbers (doubles) between 0.0 and 1.0:
  std::uniform_real_distribution<> distrib_real(0.0, 1.0);

  // 3. Generate and print some random numbers
  std::cout << "Uniform integers: ";
  for (int n = 0; n < 5; ++n) {
    std::cout << distrib_int(gen) << " ";
  }
  std::cout << std::endl;

  std::cout << "Uniform real numbers: ";
  for (int n = 0; n < 5; ++n) {
    std::cout << distrib_real(gen) << " ";
  }
  std::cout << std::endl;

  return 0;
}

## Bonus: Xavier, and  
