#!/bin/csh


nmrPipe -in ./test.ft1  \
#| nmrPipe  -fn ZF -size 256                           \
| nmrPipe  -fn FT                                 \
   -verb -ov -out test.ft2

