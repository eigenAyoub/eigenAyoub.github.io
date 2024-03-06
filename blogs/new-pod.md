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
* [Frequent problems]()
	* Copy files between difference hosts over ssh
	* Multiple remote git 
	* Local git repo ahead of remote repo, and can;t push (exceeded 100Mb limit)

---
# Essential:


```bash
# manual / pip
$ unminimize
$ apt install python3-pip

# required for `pyenv`
$ apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl

# required for `lm-eval` (there
$ apt install lzma liblzma-dev libbz2-dev   

# needed to transfer files between diff. hosts
$ apt install rsync

$ apt update && apt upgrade -y

#not needed anymore, I guess? Cuz I am using pyenv for everything?
#apt-get install software-properties-common
#add-apt-repository ppa:deadsnakes/ppa
```

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

**TODO**

---
# Python, copy pasta please.

```bash
# numpy is king,  always first and alone.
$ pip install numpy 
$ pip install matplotlib
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
$ pip install ipython
$ pip install einops # cuz you need it to learn it!
$ pip install transformers datasets  tiktoken  wandb tqdm 
# no sklearn? you zoomer.
```

---
# .dotfiles:
(would you please update this frequently?)

## VIM

```bash
set wrap
set number relativenumber

set mouse=a
set so=15
set ai
set si

set tabstop=4
set shiftwidth=4
set smarttab

autocmd FileType markdown setlocal spell
```

## tmux

```bash
################################# Basics
set -g mouse on

# main key
unbind C-b
set-option -g prefix C-Space
bind-key C-Space send-prefix

# Get the colors work
set -g default-terminal "screen-256color"
set -ga terminal-overrides ",xterm-256color:Tc"


#shift alt, switch between keys
bind -n M-H previous-window
bind -n M-L next-window

################################## Copy Pasta
set -s set-clipboard on


# Use vim keybindings in copy mode
setw -g mode-keys vi
unbind -T copy-mode-vi MouseDragEnd1Pane

# Clear selection on single click
bind -T copy-mode-vi MouseDown1Pane send-keys -X clear-selection \; select-pane

bind-key -T copy-mode-vi v send-keys -X begin-selection
bind-key -T copy-mode-vi C-v send-keys -X rectangle-toggle
bind-key -T copy-mode-vi y send-keys -X copy-selection-and-cancel

# you know exactly what this is about.
bind '"' split-window -v -c "#{pane_current_path}"
bind % split-window -h -c "#{pane_current_path}"

################################## Plugins
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-sensible'
set -g @plugin 'christoomey/vim-tmux-navigator'
set -g @plugin 'tmux-plugins/tmux-yank'

run '~/.tmux/plugins/tpm/tpm'
```	

## VS Code

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

> Local to Remote: `rsync [OPTION]... -e ssh [SRC]... [USER@]HOST:DEST`

> Remote to Local: `rsync [OPTION]... -e ssh [USER@]HOST:SRC... [DEST]`



* Local git ahead of remote one. Can't push because of some file that staged a few commits ago, but I deleted the file. Long story, but if you know you know.

* Debug something in **ipython**: `%run script.py`
* Free your the gpu memory that you have used `torch.cuda.empty_cache()`


Comments from https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/3

*  If after calling it, you still have some memory that is used, that means that you have a python variable (either torch Tensor or torch Variable) that reference it, and so it cannot be safely released as you can still access it.

* So any variable that is no longer reference is freed in the sense that its memory can be used to create new tensors, but this memory is not released to the os (so will still look like it’s used using nvidia-smi).
empty_cache forces the allocator that pytorch uses to release to the os any memory that it kept to allocate new tensors, so it will make a visible change while looking at nvidia-smi, but in reality, this memory was already available to allocate new tensors.

---
# What is the difference?

* Difference between interactive shells and login shells?

* Difference between `apt` and `apt-get`:  Uhhm, `apt` is the new `apt-get`, just use it dude, trust me!.  [Ask Ubuntu Link](https://askubuntu.com/questions/445384/what-is-the-difference-between-apt-and-apt-get)

