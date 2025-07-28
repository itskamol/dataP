#!/bin/bash
# Bash completion for file processing CLI

_cli_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Main options
    opts="--config --file1 --file2 --file1-type --file2-type --delimiter1 --delimiter2
          --matching-type --threshold --algorithm --normalize --case-sensitive
          --output --format --output-dir --include-unmatched --prefix1 --prefix2
          --batch --config-dir --parallel --chunk-size --memory-limit
          --interactive --verbose --quiet --progress
          --generate-config --template --list-templates --validate-config
          --list-files --show-columns --examples --generate-completion
          --log-level --log-file --help --version"

    case "${prev}" in
        --config|--file1|--file2|--show-columns)
            COMPREPLY=( $(compgen -f "${cur}") )
            return 0
            ;;
        --config-dir|--output-dir)
            COMPREPLY=( $(compgen -d "${cur}") )
            return 0
            ;;
        --file1-type|--file2-type)
            COMPREPLY=( $(compgen -W "csv json auto" -- "${cur}") )
            return 0
            ;;
        --matching-type)
            COMPREPLY=( $(compgen -W "one-to-one one-to-many many-to-one many-to-many" -- "${cur}") )
            return 0
            ;;
        --algorithm)
            COMPREPLY=( $(compgen -W "exact fuzzy phonetic mixed" -- "${cur}") )
            return 0
            ;;
        --format)
            COMPREPLY=( $(compgen -W "csv json both" -- "${cur}") )
            return 0
            ;;
        --template)
            COMPREPLY=( $(compgen -W "basic_csv csv_to_json uzbek_text high_precision bulk_processing" -- "${cur}") )
            return 0
            ;;
        --progress)
            COMPREPLY=( $(compgen -W "none simple detailed" -- "${cur}") )
            return 0
            ;;
        --log-level)
            COMPREPLY=( $(compgen -W "DEBUG INFO WARNING ERROR CRITICAL" -- "${cur}") )
            return 0
            ;;
        --generate-completion)
            COMPREPLY=( $(compgen -W "bash zsh fish" -- "${cur}") )
            return 0
            ;;
    esac

    COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
    return 0
}

complete -F _cli_completion cli.py
complete -F _cli_completion python3\ cli.py
