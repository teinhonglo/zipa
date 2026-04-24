#!/bin/bash

file_list="phn_text transcript_phn_text wav.scp wrd_text"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <dest-data-dir> <src-data-dir1> <src-data-dir2>"
    exit 1
fi

echo "$0 $@"  # Print the command line for logging

. ./path.sh
. ./local/parse_options.sh

dest=$1;
shift;

first_src=$1;

rm -r $dest 2>/dev/null || true
mkdir -p $dest;
chmod 777 $dest;

# rm dest file
for file in $file_list; do
    [ -f $dest/$file ] && rm $dest/$file
    touch $dest/$file
done

# cat file to dest file
for dir in $*; do
    for file in $file_list; do
        cat $dir/$file >> $dest/$file
    done
done
