#!/bin/bash

#train=$(perl -e 'for($i=0; $i<=18; $i++){printf("%02d\n", $i)}')
#rm train.pos
#touch train.pos
#for id in $train; do
#    for f in `ls $id/*.POS`; do 
#        python ./wsj_convert.py $f >> train.pos;
#    done
#done

devel=$(perl -e 'for($i=19; $i<=21; $i++){printf("%02d\n", $i)}')
rm devel.pos
touch devel.pos
for id in $devel; do
    for f in `ls $id/*.POS`; do
        python ./wsj_convert.py $f >> devel.pos;
    done
done

#tt=$(perl -e 'for($i=22; $i<=24; $i++){printf("%02d\n", $i)}')
#rm test.pos
#touch test.pos
#for id in $tt; do
#    for f in `ls $id/*.POS`; do
#        python ./wsj_convert.py $f >> test.pos;
#    done
#done
