#!/bin/bash

map_file=local/timit/phones.60-48-39.l2arctic.map # *.l2arctic.map / *.timit.map

. path.sh
. ./local/parse_options.sh

if [ $# -ne 3 ]; then
   echo "$0 <timit-corpus> <phoneme_map> <data-kaldi/timit>"
   exit 1;
fi

phoneme_map=$2
timit_dir=$3

sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi

train_dir=train
test_dir=test

tmpdir=`pwd`/tmp

[ ! -d $tmpdir ] && mkdir -p $tmpdir
[ ! -d $timit_dir ] && mkdir -p $timit_dir

for data in train dev test; do

  [ ! -d $timit_dir/$data ] && mkdir -p $timit_dir/${data}
  # get train / dev / test flist
  # use si and sx only
  find $1/{$train_dir,$test_dir} -not \( -iname 'SA*' \) -iname '*.WAV' \
    | grep -f local/timit/${data}_spk > $tmpdir/timit_${data}_sph.flist

  # get utt id
  sed -e 's:.*/\(.*\)/\(.*\).WAV$:\1_\2:i' $tmpdir/timit_${data}_sph.flist \
    > $tmpdir/timit_${data}_sph.uttids

  # wav.scp
  paste -d" " $tmpdir/timit_${data}_sph.uttids $tmpdir/timit_${data}_sph.flist \
    | sort -k1,1 > $timit_dir/${data}/wav.scp
    
  awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < $timit_dir/${data}/wav.scp > $timit_dir/${data}/wav_sph.scp

  for y in wrd phn; do
    # use si and sx only
    find $1/{$train_dir,$test_dir} -not \( -iname 'SA*' \) -iname '*.'$y'' \
        | grep -f local/timit/${data}_spk > $tmpdir/timit_${data}_txt.flist
    sed -e 's:.*/\(.*\)/\(.*\).'$y'$:\1_\2:i' $tmpdir/timit_${data}_txt.flist \
        > $tmpdir/timit_${data}_txt.uttids
    while read line; do
        [ -f $line ] || error_exit "Cannot find transcription file '$line'";
        cut -f3 -d' ' "$line" | tr '\n' ' ' | sed -e 's: *$:\n:'
    done < $tmpdir/timit_${data}_txt.flist > $tmpdir/timit_${data}_txt.trans

    # uttid, text
    paste -d" " $tmpdir/timit_${data}_txt.uttids $tmpdir/timit_${data}_txt.trans \
        | sort -k1,1 > $tmpdir/timit_${data}.trans

    # text
    cat $tmpdir/timit_${data}.trans | sort > $timit_dir/${data}/${y}_text || exit 1;
    if [ $y == phn ]; then
        cp $timit_dir/${data}/${y}_text $timit_dir/${data}/${y}_text.tmp
        python local/utils/normalize_phone.py --map $map_file --to $phoneme_map --src $timit_dir/${data}/${y}_text.tmp --tgt $timit_dir/${data}/${y}_text
        rm -f $timit_dir/${data}/${y}_text.tmp
        cp $timit_dir/${data}/${y}_text $timit_dir/${data}/transcript_${y}_text
    fi
  done
done

rm -rf $tmpdir

echo "Data preparation succeeded"
