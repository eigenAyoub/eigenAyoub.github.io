---
layout: post
title:
permalink: /blogs/vim/
---


### [My number 1 rule](https://www.youtube.com/watch?v=XDdDQQ8uLhY):
> Something in your VIM workflow that sounds inefficient? too long? $$\implies$$Then yes, you're right, and there is a trick within VIM that solves it quickly and efficiently. Google it, trust me.


### My favorite hacks:
* [Switch your `<ESC>` and `<CAPS>` keys](#hack-1)
* [Disable your arrow keys](#hack-2)
* [Make your visual search](#hack-3)
* [Faster navigation between splits inside VIM](#hack-4)
* [Move between local visual lines](#hack-5)
* [Embrace the power of  Registers 0, 1](#hack-6)
* [Ditch w for W](#hack-7)
* [Quickly source your files from VIM](#hack-8)

### Content:

#### **Switch your `<ESC>` and `<CAPS>` keys** <a name="hack-1"></a>
* How? Add the following to your **.bashrc**:    
    * `setxkbmap -option caps:swapescape`


#### **Disable your arrow keys:**<a name="hack-2"></a>
If you are serious about learning VIM, then please disable your arrow keys.
* How: Add the following to your **.vimrc**:
    * `noremap <Up> <Nop>`
    * `noremap <Down> <Nop>`
    * `noremap <Left> <Nop>`
    * `noremap <Right> <Nop>`
    * `inoremap <Left>  <NOP>`
    * `inoremap <Right> <NOP>`
    * `inoremap <Up>    <NOP>`
    * `inoremap <Down>  <NOP>`

**P.S: "If you're really about that life"**
> You should only use`<H-J-K-L>` for local changes/movements,  and use VIM advanced motions for big jumps. I personally don't think I'm there yet.


#### **Let the cursor move to your search pattern while typing:**  <a name="hack-3"></a>
Search in VIM is usually handled by `/` and `?`, one drawback is that you have type the pattern you're looking for then press `<ENTER>` to move to your target. `incsearch` let you move to your target while typing. Add the following to your **.vimrc**:
* `set incsearch`

#### **Faster navigation between splits inside VIM:**  <a name="hack-4"></a>
In your **.vimrc** add the following:
*  `nnoremap <C-J> <C-W><C-J>` 
*  `nnoremap <C-K> <C-W><C-K>`                                                       
*  `nnoremap <C-L> <C-W><C-L>`                                                       
*  `nnoremap <C-H> <C-W><C-H>`

#### **Move between local visual lines:**  <a name="hack-5"></a>
VIM splits a long line into multiple "visual" lines, yet, `<h,j,k,l>`, still jumps the whole line. If you want move vertically through the virtual lines, the you can use `gi` and `gk`. I personally have the keys `j` and `k` remaped indefinetely as follows:
* `noremap j gj`
* `noremap k gk`


#### **Embrace the power of  Registers 0, 1:**  <a name="hack-6"></a>
If you just started using VIM, then you might face this situation every damn day:
1. You yank the word w1
2. You move to another word w2
3. You delete w2
4. Click on `p` (in your mind, you wish to paste the word w1)
5. VIM yanks the word w2 instead. 
6. You should normally start swearing at VIM.

The hack is to start embracing the world of REGISTERS. It's okay if you don't want to use them for general purposes (MACROS), but you should know that **Register 0** is your friend. It holds the last yanked thing.  Which you can quickly access using **`"0p`**.

#### **Ditch w,b,e for W, B and E**  <a name="hack-7"></a>
Most of the time, I find myself wanting to use a motion or action on a "Big-Word". Which you can access using `W` (resp. `E`, and `B`) instead of `w` (resp. `e`, and `b`). What I refered to as a "Big-Word" is the concatenation of any non-empty/ non-whitespace characters.

#### **Quickly source your .vimrc from VIM**<a name="hack-8"></a>
Type the following in normal mode, duuh:  `:so%`

## TO-DO:
* REgisters?
* Plugins?
* cp paste between files?
