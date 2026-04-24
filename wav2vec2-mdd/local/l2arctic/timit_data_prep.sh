#!/bin/bash

if [ $# -ne 3 ]; then
   echo "$0 <timit-corpus> <phoneme_map> <data-kaldi/timit-train>"
   exit 1;
fi

map_file=local/l2arctic/phones.60-48-39.map

. path.sh
. ./local/parse_options.sh

phoneme_map=$2
timit_train_dir=$3

sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi

train_dir=train
test_dir=test

tmpdir=`pwd`/tmp

[ ! -d $tmpdir ] && mkdir -p $tmpdir
[ ! -d $timit_train_dir ] && mkdir -p $timit_train_dir


ls -d "$1"/*/DR*/* | sed -e "s:^.*/::" > $timit_train_dir/timit_train_spk.list

# use si and sx
find $1/{$train_dir,$test_dir}  -iname '*.WAV' \
  | grep -f $timit_train_dir/timit_train_spk.list > $tmpdir/timit_train_sph.flist

# get utt id
sed -e 's:.*/\(.*\)/\(.*\).WAV$:\1_\2:i' $tmpdir/timit_train_sph.flist \
  > $tmpdir/timit_train_sph.uttids

# wav.scp
paste -d" " $tmpdir/timit_train_sph.uttids $tmpdir/timit_train_sph.flist \
  | sort -k1,1 > $timit_train_dir/wav.scp

awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < $timit_train_dir/wav.scp > $timit_train_dir/wav_sph.scp

for y in wrd phn; do
  find $1/{$train_dir,$test_dir}  -iname '*.'$y'' \
      | grep -f $timit_train_dir/timit_train_spk.list > $tmpdir/timit_train_txt.flist
  sed -e 's:.*/\(.*\)/\(.*\).'$y'$:\1_\2:i' $tmpdir/timit_train_txt.flist \
      > $tmpdir/timit_train_txt.uttids
  while read line; do
      [ -f $line ] || error_exit "Cannot find transcription file '$line'";
      cut -f3 -d' ' "$line" | tr '\n' ' ' | sed -e 's: *$:\n:'
  done < $tmpdir/timit_train_txt.flist > $tmpdir/timit_train_txt.trans

  # uttid, text
  paste -d" " $tmpdir/timit_train_txt.uttids $tmpdir/timit_train_txt.trans \
      | sort -k1,1 > $tmpdir/timit_train.trans

  # text
  cat $tmpdir/timit_train.trans | sort > $timit_train_dir/${y}_text || exit 1;
  if [ $y == phn ]; then
      cp $timit_train_dir/${y}_text $timit_train_dir/${y}_text.tmp
      python local/utils/normalize_phone.py --map $map_file --to $phoneme_map --src $timit_train_dir/${y}_text.tmp --tgt $timit_train_dir/${y}_text
      rm -f $timit_train_dir/${y}_text.tmp
      cp $timit_train_dir/${y}_text $timit_train_dir/transcript_${y}_text
  fi
done

rm -rf $tmpdir

echo "Data preparation succeeded"
