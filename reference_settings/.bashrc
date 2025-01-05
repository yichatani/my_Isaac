# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

# don't put duplicate lines or lines starting with space in the history.
# See bash(1) for more options
HISTCONTROL=ignoreboth

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# If set, the pattern "**" used in a pathname expansion context will
# match all files and zero or more directories and subdirectories.
#shopt -s globstar

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color|*-256color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
#force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
	# We have color support; assume it's compliant with Ecma-48
	# (ISO/IEC-6429). (Lack of such support is extremely rare, and such
	# a case would tend to support setf rather than setaf.)
	color_prompt=yes
    else
	color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*)
    ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# colored GCC warnings and errors
#export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# Add an "alert" alias for long running commands.  Use like so:
#   sleep 10; alert
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

##########>>>>>>>>>> ISAAC_PYTHON_PATH

export ISAAC_SIM_PATH=/home/ani/.local/share/ov/pkg/isaac-sim-4.2.0
export PYTHONPATH=$PYTHONPATH:$ISAAC_SIM_PATH/kit/python/lib/python3.10/site-packages:$ISAAC_SIM_PATH/python_packages:$ISAAC_SIM_PATH/exts/omni.isaac.kit:$ISAAC_SIM_PATH/kit/kernel/py:$ISAAC_SIM_PATH/kit/plugins/bindings-python:$ISAAC_SIM_PATH/exts/omni.isaac.lula/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.exporter.urdf/pip_prebundle:$ISAAC_SIM_PATH/extscache/omni.kit.pip_archive-0.0.0+10a4b5c0.lx64.cp310/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.isaac.core_archive/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.isaac.ml_archive/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.pip.compute/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.pip.cloud/pip_prebundle

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ISAAC_SIM_PATH/kit:$ISAAC_SIM_PATH/kit/kernel/plugins:$ISAAC_SIM_PATH/kit/libs/iray:$ISAAC_SIM_PATH/kit/plugins:$ISAAC_SIM_PATH/kit/plugins/bindings-python:$ISAAC_SIM_PATH/kit/plugins/carb_gfx:$ISAAC_SIM_PATH/kit/plugins/rtx:$ISAAC_SIM_PATH/kit/plugins/gpu.foundation:$ISAAC_SIM_PATH/exts/omni.usd.schema.isaac/plugins/IsaacSensorSchema/lib:$ISAAC_SIM_PATH/exts/omni.usd.schema.isaac/plugins/RangeSensorSchema/lib:$ISAAC_SIM_PATH/exts/omni.isaac.lula/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.exporter.urdf/pip_prebundle

# Self Added Isaac python path
export PYTHONPATH=$PYTHONPATH:$ISAAC_SIM_PATH/exts/omni.isaac.motion_generation

# Isaac lab related
export ISAACSIM_DIR="/home/ani/miniconda3/envs/anygrasp/bin/isaacsim"
export ISAACLAB_DIR="/home/ani/IsaacLab"

export EXP_PATH="/home/ani/.local/share/ov/pkg/isaac-sim-4.2.0/apps"
export CARB_APP_PATH="/home/ani/.local/share/ov/pkg/isaac-sim-4.2.0/kit"
export ISAAC_PATH="/home/ani/.local/share/ov/pkg/isaac-sim-4.2.0"
#source /home/ani/.local/share/ov/pkg/isaac-sim-4.2.0/setup_conda_env.sh
##########<<<<<<<<<<

# >>> fishros initialize >>>
source /opt/ros/humble/setup.bash
# <<< fishros initialize <<<

eval "$(/home/ani/miniconda3/bin/conda shell.bash hook)"
#alias activate_conda_env='source ~/miniconda3/bin/activate'
conda config --set auto_activate_base false
