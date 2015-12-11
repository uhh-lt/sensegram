# Created by newuser for 4.3.17

# The following lines were added by compinstall

zstyle ':completion:*' auto-description 'specify: %d'
zstyle ':completion:*' completer _oldlist _expand _complete _ignored _match _approximate _prefix
zstyle ':completion:*' completions 1
zstyle ':completion:*' condition 0
zstyle ':completion:*' file-sort name
zstyle ':completion:*' format 'format: %d'
zstyle ':completion:*' glob 1
zstyle ':completion:*' group-name ''
zstyle ':completion:*' insert-unambiguous true
zstyle ':completion:*' list-colors ''
zstyle ':completion:*' list-prompt %SAt %p: Hit TAB for more, or the character to insert%s
zstyle ':completion:*' match-original both
zstyle ':completion:*' matcher-list '' '+l:|=* r:|=*' '+r:|[._-]=** r:|=**' '+m:{[:lower:][:upper:]}={[:upper:][:lower:]}'
zstyle ':completion:*' max-errors 5 numeric
zstyle ':completion:*' menu select=1
zstyle ':completion:*' original true
zstyle ':completion:*' preserve-prefix '//[^/]##/'
zstyle ':completion:*' prompt '%e'
zstyle ':completion:*' select-prompt %SScrolling active: %l current selection at %p%s
zstyle ':completion:*' special-dirs true
zstyle ':completion:*' substitute 1
zstyle ':completion:*' use-compctl true
zstyle :compinstall filename '/home/pi/.zshrc'

autoload -Uz compinit
compinit
# End of lines added by compinstall
# Lines configured by zsh-newuser-install
HISTFILE=~/.histfile
HISTSIZE=2000
SAVEHIST=2000
setopt appendhistory autocd extendedglob
unsetopt beep
bindkey -e
# End of lines configured by zsh-newuser-install

# {{{ colors
local BLACK="%{"$'\033[01;30m'"%}"
local GREEN="%{"$'\033[01;32m'"%}"
local RED="%{"$'\033[01;31m'"%}"
local YELLOW="%{"$'\033[01;33m'"%}"
local BLUE="%{"$'\033[01;34m'"%}"
local CYAN="%{"$'\033[01;36m'"%}"
local BOLD="%{"$'\033[01;39m'"%}"
local NONE="%{"$'\033[00m'"%}"
# }}}


# prompt
export PS1="${BLUE}%n${NONE}@${GREEN}%m ${CYAN}%~ ${NONE}%% "
[[ $UID = 0 ]] && export PS1="${RED}%n${NONE}@${BLUE}%m ${CYAN}%~ ${NONE}%% "

# FUNCTION

ec(){
    emacsclient -c $1 &> /dev/null &
}


# ALIAS

alias l='ls -aghX --classify --group-directories-first --color=always'
alias grep='grep --color=always --line-number --initial-tab --binary-files=without-match --recursive'
alias emacs='emacs --daemon'

alias py='source .env/bin/activate'

alias b='cd ..'
alias bb='cd ../..'
alias bbb='cd ../../..'
alias bbbb='cd ../../../..'

alias ke="emacsclient -e '(kill-emacs)'"