#!/bin/csh

nmrPipe -in ./test.fid  \
| nmrPipe  -fn SOL \
| nmrPipe  -fn SP -off 0.5 -end 0.95 -pow 2  -c 0.5  \
#| nmrPipe  -fn ZF -auto                            \
| nmrPipe  -fn FT -auto                                 \
| nmrPipe  -fn PS -p0 0.0 -p1 0.0 -di -verb       \
| nmrPipe  -fn EXT -x1 11.0ppm -xn 6.0ppm -sw \
| nmrPipe  -fn TP                                   \
| nmrPipe  -fn TRI -loc 1 -lHi 0.4 -rHi 0 \
| nmrPipe  -fn PS -p0 0 -p1 0               \
   -verb -ov -out test.ft1
