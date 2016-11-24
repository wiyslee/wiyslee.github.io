#!/bin/bash

# Arguments
while [[ $# -gt 1 ]]
do
	key="$1"

	case $key in
	    -l|--latex)
	    LATEX="$2"
	    shift # past argument
	    ;;
	    -m|--markdown)
	    MD="$2"
	    shift # past argument
	    ;;
	    --default)
	    DEFAULT=YES
	    ;;
	    *)
	            # unknown option
	    ;;
	esac
	shift # past argument or value
done

# Convert using pandoc
pandoc -f latex -t markdown -o ${MD} ${LATEX} | python ~/Projects/wiyslee.github.io/latex-to-md.py --m ${MD}

#| sed -e 's/\([^\$]\)\(\$\)\([^\$]\)/\1\$\$\3/g' | sed -e 's/^\(\$\)\([^\$]\)/\$\$\2/g'
# > ${MD}
