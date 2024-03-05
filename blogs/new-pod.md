---
layout: default
title:
permalink: /blogs/container/
---


Quick reset to my compute pod.

---
# Jump to it:
* [Essential]()
* [Pyenv]()
* [Git]()
* [pip it]()
* [dotfiles]()

---
# Essential:


```bash
# manual / pip
$ unminimize
$ apt install python3-pip

# required for `pyenv`
$ apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl

# required for `lm-eval` (there
$ apt-get install lzma liblzma-dev libbz2-dev   

$ apt update && apt upgrade -y

#not needed anymore, I guess? Cuz I am using pyenv for everything?
#apt-get install software-properties-common
#add-apt-repository ppa:deadsnakes/ppa
```
Additional stuff:
`$ apt install rsync`
---
# Pyenv:

* Run the following installer:
```bash
$ curl https://pyenv.run | bash
```

* Add the following to your `.bash_profile`:

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```
* Add the following to your `.bashrc`:

```bash
eval "$(pyenv virtualenv-init -)"
```

* *Game time:* what you need to know.

```bash
pyenv install 3.12  #to install
pyenv versions      #to list all available versions
pyenv global 3.12   #to use v 3.12 
```

---
# git:

**Update Configs:**

```bash
git config --global user.name  eigenAyoub
git config --global user.email benayad7@outlook.com
```

**SSH**
1. Generate key:
	* `ssh-keygen -t ed25519 -C "benaya7@outlook.com"`
2. Start `ssh-agent`, then add private key to it:
	* `eval "$(ssh-agent -s)` (this starts the agent)
	* `ssh-add ~/.ssh/id_ed25519`
3. Add public key to your remote git server.
	* `cat ~/.ssh/id_ed25519.pub`
	*  Paste it in github/gitlab.

**Set up multiple remotes**
TODO

---
# Python, copy pasta please.

```bash
# numpy is king, so always install it alone ;)
$ pip install numpy 
$ pip install matplotlib
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
$ pip install ipython
$ pip install einops # cuz you need it to learn it!
$ pip install transformers datasets  tiktoken  wandb tqdm 
```

---
# .dotfiles:

* **.bashrc**
Not sure why I had this before?

 On your way, add to your `.bashrc`:
`~~> alias ls='ls --color=auto'`

* **.vimrc**

* **.tmux.conf**

```bash
set -g mouse on
```	

* vscode setting:

```json
{
    "security.workspace.trust.untrustedFiles": "open",
    "window.zoomLevel": 1,
    "vim.insertModeKeyBindingsNonRecursive": [
        {
            "before": ["<ESC>"],
            "after": ["<ESC>"],
            "commands": [
                "workbench.action.files.save"
            ]
        }

    ],
    "keyboard.dispatch": "keyCode",
    "vim.normalModeKeyBindingsNonRecursive": [
        {
            "before": ["Z", "Z"],
            "commands": [":w"]
        },
        {
            "before": ["g", "p", "d"],
            "commands": ["editor.action.peekDefinition"]
        }
    ],
    "vim.smartRelativeLine": true,
    "editor.cursorSurroundingLines": 8,
    "vim.useSystemClipboard": true,
    "glassit.alpha": 220,
    "editor.minimap.enabled": false,
}
```

	
---
# Problems and Fixes:

* Copy files from one pod to another using `rsync`:
Syntax:

Local to Remote: `rsync [OPTION]... -e ssh [SRC]... [USER@]HOST:DEST`
Remote to Local: `rsync [OPTION]... -e ssh [USER@]HOST:SRC... [DEST]`


Run the minimal GPU/pytorch script (gpu-test.py available in the .config), make sure everything is [Operating Smoothly](https://www.youtube.com/watch?v=4TYv2PhG89A).


Setting python:
`python3 -m site`
> I want to start using type hinting with python, and I'm fed up with python 3.8.

1. Local git ahead of remote one. Can't push because of some file that staged a few commits ago, but I deleted the file. Long story, but if you know you know.

2. Debug something in **ipython**: `%run script.py`
* Free your the gpu memory that you have used `torch.cuda.empty_cache()`


Comments from https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/3

	*  If after calling it, you still have some memory that is used, that means that you have a python variable (either torch Tensor or torch Variable) that reference it, and so it cannot be safely released as you can still access it.
	* So any variable that is no longer reference is freed in the sense that its memory can be used to create new tensors, but this memory is not released to the os (so will still look like it’s used using nvidia-smi).
empty_cache forces the allocator that pytorch uses to release to the os any memory that it kept to allocate new tensors, so it will make a visible change while looking at nvidia-smi, but in reality, this memory was already available to allocate new tensors.
