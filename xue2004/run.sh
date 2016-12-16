#!/bin/bash

#./xue.py learn -s data/conll05st-release/train/synt.gold/ \
#    -p data/conll05st-release/train/props/ \
#    -w data/conll05st-release/train/words2/ \
#    -m xue2004.synt.gold.m
#
#./xue.py labeling -m xue2004.synt.gold.m \
#    -s data/conll05st-release/devel/synt.gold/ \
#    -w data/conll05st-release/devel/words2/ \
#    -t data/conll05st-release/devel/targets/ > devel.answer.gold.synt.props
#
#zcat ./data/conll05st-release/devel/props/devel.24.props.gz | ./srl-eval.pl - devel.answer.gold.synt.props

#./xue.py learn -s data/conll05st-release/train/synt.col2/ \
#    -p data/conll05st-release/train/props/ \
#    -w data/conll05st-release/train/words2/ \
#    -m xue2004.synt.col2.m
#
#./xue.py labeling -m xue2004.synt.col2.m \
#    -s data/conll05st-release/devel/synt.col2/ \
#    -w data/conll05st-release/devel/words2/ \
#    -t data/conll05st-release/devel/targets/ > devel.answer.synt.col2.props
#
#zcat ./data/conll05st-release/devel/props/devel.24.props.gz | ./srl-eval.pl - devel.answer.synt.col2.props

./xue2004.py learn -s data/conll05st-release/train/synt.cha/ \
    -p data/conll05st-release/train/props/ \
    -w data/conll05st-release/train/words2/ \
    -m xue2004.synt.cha.m \
    --simplified

./xue2004.py labeling -m xue2004.synt.cha.m \
    -s data/conll05st-release/devel/synt.cha/ \
    -w data/conll05st-release/devel/words2/ \
    -t data/conll05st-release/devel/targets/ > devel.answer.synt.cha.props

./xue2004.py labeling -m xue2004.synt.cha.m \
    -s data/conll05st-release/test.wsj/synt.cha/ \
    -w data/conll05st-release/test.wsj/words/ \
    -t data/conll05st-release/test.wsj/targets/ > test.answer.synt.cha.props

zcat ./data/conll05st-release/devel/props/devel.24.props.gz | ./srl-eval.pl - devel.answer.synt.cha.props
zcat ./data/conll05st-release/test.wsj/props/test.wsj.props.gz | ./srl-eval.pl - test.answer.synt.cha.props
