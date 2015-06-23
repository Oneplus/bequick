#!/bin/bash

./xue2004.py labeling -m xue2004.m \
    -s data/conll05st-release/devel/synt.gold/ \
    -w data/conll05st-release/devel/words2/ \
    -t data/conll05st-release/devel/targets/ > devel.answer.props

zcat ./data/conll05st-release/devel/props/devel.24.props.gz | ./srl-eval.pl - devel.answer.props
