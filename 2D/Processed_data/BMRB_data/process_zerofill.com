#!/bin/csh -f


echo '|   Processing time domain MDD reconstruction '
echo
echo Processing YZ dimensions
xyz2pipe -in ./n15_noesy.ft1           \
| nmrPipe  -fn SP -off 0.5 -end 1 -pow 2 -c 0.5  \
| nmrPipe  -fn ZF -size 64                             \
#| nmrPipe  -fn FT                                   \
| nmrPipe  -fn PS -p0  0 -p1  0               \
| nmrPipe  -fn ZTP \
| nmrPipe  -fn SP -off 0.5 -end 1 -pow 2 -c 0.5  \
| nmrPipe  -fn ZF -size 64                          \
#| nmrPipe  -fn FT                                   \
| nmrPipe  -fn PS -p0  0 -p1  0                 \
| nmrPipe  -fn ZTP \
| nmrPipe  -fn TP \
  > ./fid_temp_ZF.dat

exit
    
