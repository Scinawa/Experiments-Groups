#!/bin/sh

rm -rf original.txt
rm -rf corre2.txt
rm -rf corre3.txt


#count=10
#for i in $(seq $count); do
#    python3 main.py >> original.txt
#done

count=10
for i in $(seq $count); do
    python3 extended-main.py --correlation 3 >> corre3.txt
    echo "done $i"
done


for i in $(seq $count); do
    python3 extended-main.py --correlation 2 >> corre2.txt
done
