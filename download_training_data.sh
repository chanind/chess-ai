#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")"
DATADIR=./data
mkdir -p $DATADIR
cd $DATADIR

# misc lichess games
wget -O lichess_db_standard_rated_2016-09.pgn.bz2 https://database.lichess.org/standard/lichess_db_standard_rated_2016-09.pgn.bz2
bzip2 -d lichess_db_standard_rated_2016-09.pgn.bz2