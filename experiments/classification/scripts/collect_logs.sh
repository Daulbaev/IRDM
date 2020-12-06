#!/bin/bash

mkdir -p $2
cd $1

for file in `find . ! -name '*.pth' | sed 's/^.\///'`
do
	echo $file

        if [ -d "./$file" ];
        then
                mkdir -p "$2/$file"
        else
                cp "./$file" "$2/$file"
        fi
done

### bash collect_logs.sh  /workspace/raid/data/jgusak/anode/  /workspace/raid/data/jgusak/anode_logs/

