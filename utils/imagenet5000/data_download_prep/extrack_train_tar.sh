#!/bin/bash
for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  (cd $d && tar xvf ../$f && rm ../$f)
done
