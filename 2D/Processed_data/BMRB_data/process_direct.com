#!/bin/csh -f


echo '|   Processing time domain MDD reconstruction '
echo
echo Processing YZ dimensions
xyz2pipe -in data/test%03d.fid -x -verb              \
| nmrPipe  -fn POLY -time                           \
| nmrPipe  -fn SP -size 512 -off 0.50 -end 1 -pow 2 -c 0.5  \
| nmrPipe  -fn ZF -auto                             \
| nmrPipe  -fn FT -auto                                  \
| nmrPipe  -fn PS -p0 43 -p1 0.0 -di              \
| nmrPipe  -fn EXT -x1 11.0ppm -xn 6.0ppm -sw -verb \
#| nmrPipe  -fn POLY -auto -ord 0                    \
| pipe2xyz -out data/test%03d.ft1 -x -ov -verb      \

xyz2pipe  -in data/test%03d.ft1 -y -verb            \
  > ./n15_noesy.ft1


exit
    
