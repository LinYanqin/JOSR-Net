#!/bin/csh

nmrPipe -in ./test.fid  \
| nmrPipe  -fn SOL \
| nmrPipe  -fn SP -off 0.5 -end 1 -pow 1  -c 0.5  \
#| nmrPipe  -fn ZF -auto                            \
| nmrPipe  -fn FT -auto                                 \
| nmrPipe  -fn PS -p0 -58 -p1 0.0 -di -verb       \
| nmrPipe  -fn EXT -x1 11.0ppm -xn 6.0ppm -sw \
| nmrPipe  -fn TP                                   \
| nmrPipe  -fn SP -off 0.5 -end 1 -pow 1 -c 0.5  \
| nmrPipe  -fn PS -p0 0 -p1 0               \
   -verb -ov -out test.ft1
