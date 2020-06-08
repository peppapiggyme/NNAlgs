#!/bin/bash

TARGZ_FILE=saved-$(date +%Y%m%d).tar.gz
tar -zcvf ${TARGZ_FILE} saved/
rsync -rP --rsh=ssh ${TARGZ_FILE} zhangbw@atlasui01.ihep.ac.cn:/scratchfs/atlas/zhangbw/fromMWT2/
