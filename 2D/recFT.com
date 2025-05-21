#!/bin/csh -f


echo '|   Processing time domain MDD reconstruction '
echo
echo Processing YZ dimension
set I = 1
echo $I
set FID_file = ./nmrpipe_data/res_full.dat
xyz2pipe -in $FID_file            \
| nmrPipe  -fn TP -auto           \
| nmrPipe  -fn ZF -size 128		\
| nmrPipe  -fn SP -off 0.5 -end 1 -pow 2 -c 0.5 \
| nmrPipe  -fn FT          \
| nmrPipe  -fn PS -p0 0.0 -p1 -0.0  -di\
| nmrPipe  -fn TP  -auto                            \
| nmrPipe  -fn ZTP                                  \
| nmrPipe  -fn ZF -size 128		  \
| nmrPipe  -fn SP -off 0.5 -end 1 -pow 2 -c 0.5 \
| nmrPipe  -fn FT -alt                   \
| nmrPipe  -fn PS -p0  0.0 -p1 0 -di\
| nmrPipe  -fn ZTP   \
| nmrPipe  -fn TP -auto           \
| pipe2xyz -out ./nmrpipe_data/resCN3D.dat -x -verb -ov
proj3D.tcl -in ./nmrpipe_data/resCN3D.dat -abs

nmrPipe -in 15N.13C.dat                          \
| nmrPipe  -ov -out  ./nmrpipe_data/resCN.dat

set FID_file = ./nmrpipe_data/label_3D.dat
xyz2pipe -in $FID_file            \
| nmrPipe  -fn TP -auto           \
| nmrPipe  -fn ZF -size 128		\
| nmrPipe  -fn SP -off 0.5 -end 1 -pow 2 -c 0.5 \
| nmrPipe  -fn FT          \
| nmrPipe  -fn PS -p0 0.0 -p1 -0.0  -di\
| nmrPipe  -fn TP  -auto                            \
| nmrPipe  -fn ZTP                                  \
| nmrPipe  -fn ZF -size 128		     \
| nmrPipe  -fn SP -off 0.5 -end 1 -pow 2 -c 0.5 \
| nmrPipe  -fn FT -alt                   \
| nmrPipe  -fn PS -p0  0.0 -p1 0 -di\
| nmrPipe  -fn ZTP   \
| nmrPipe  -fn TP -auto           \
| pipe2xyz -out ./nmrpipe_data/label3D.dat -x -verb -ov
proj3D.tcl -in ./nmrpipe_data/label3D.dat -abs

nmrPipe -in 15N.13C.dat                          \
| nmrPipe  -ov -out  ./nmrpipe_data/labelCN.dat
