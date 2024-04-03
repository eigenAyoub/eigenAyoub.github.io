---
layout: default
title:
permalink: /blogs/linux/
---

Some useful Linux day 2 day stuff that most I find most useful when managing multiple checkpoints (weights of ML models).

### **Size of each sub-directories:**

```bash
$ du -h --max-depth=1 
$ du -h --max-depth=1 | grep 'G' 
```

### **Delete all files starting with some pattern `pattern_X`:**

```bash
$ ls  | grep "^pattern_X" | xargs rm
```

### **Only view sub-directories of a current directory:**

```bash
$ ls -l | grep "^d"
```

### **View all files of certain pattern `pattern_X`, except one `pattern_XY`:**

```bash
$ ls | grep "^pattern_X" | grep -v "^pattern_XY" 
```
> Example of use: When managing checkpoints during training (weights of ML models), and you would like to delete most your checkpoints, except maybe the last one.

### **View last added files of some pattern `pattern_X`**

```bash
$ ls -t | grep "^pattern_X" | head -n 10 
```

### **A very very practical use:**

* I have a bunch of checkpoints that I need to evaluate using the script `owt_eval.py`: 

```bash
$ ls -t check2/ | grep "^256" | head -n 9 | while read -r line; do 
$ 	python owt_eval.py "./check2/$line"; 
$ done
```

* I want to use another script `kron_to_gpt.py`, I need for each checkpoint an `output_directory` that usually depends on the checkpoint iteration step.

```bash
$ ls -t check2/ | grep "^256" | head -n 9 | while read -r line; do 
$ 	number=${line:31:-3}
$ 	python kron_to_gpt.py $line $number
$ done
```

My checkpoints are all of the format `checkpoint_long_string_number.pt`, hence `${line:31:-3}` extracts the number before pt.
