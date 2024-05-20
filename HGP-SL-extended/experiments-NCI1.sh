#!/bin/sh

rm -rf original-NCI1.txt
rm -rf corre2-NCI1-so.txt
rm -rf corre3-NCI1-so.txt
rm -rf corre4-NCI1-so.txt
rm -rf corre2-NCI1-mo.txt
rm -rf corre3-NCI1-mo.txt
rm -rf corre4-NCI1-mo.txt


rm -rf original-NCI1-PR.txt
rm -rf corre2-NCI1-PR-so.txt
rm -rf corre3-NCI1-PR-so.txt
rm -rf corre4-NCI1-PR-so.txt
rm -rf corre2-NCI1-PR-mo.txt
rm -rf corre3-NCI1-PR-mo.txt
rm -rf corre4-NCI1-PR-mo.txt

rm -rf original-NCI1-M1.txt
rm -rf corre2-NCI1-M1-so.txt
rm -rf corre3-NCI1-M1-so.txt
rm -rf corre4-NCI1-M1-so.txt
rm -rf corre2-NCI1-M1-mo.txt
rm -rf corre3-NCI1-M1-mo.txt
rm -rf corre4-NCI1-M1-mo.txt


count=2

for i in $(seq $count); do
    echo "Original"
    python3 main.py --b 25 --dataset NCI1 >> original-NCI1.txt
    echo "done $i"
done

for correlation in $(seq 2 4); do
    for i in $(seq $count); do
        echo "SO corre${correlation}"
        python3 extended-main.py --b 25 --correlation ${correlation} --dataset NCI1 --read_dataset True --seed $i >> corre${correlation}-NCI1-so.txt
        echo "done $i"
    done
done


for correlation in $(seq 2 4); do
    for i in $(seq $count); do
        echo "MO corre${correlation}"
        python3 extended-main.py --b 25 --correlation ${correlation} --multi_orbits TRUE --dataset NCI1 --read_dataset True --seed $i >> corre${correlation}-NCI1-so.txt
        echo "done $i"
    done
done



########## EXPERIMENTS WITH DIFFERENT pooling ratios ##########:


for i in $(seq $count); do
    echo "Original pooling ratio 0.2"
    python3 main.py --b 25 --dataset NCI1 --pooling_ratio 0.2 >> original-NCI1-PR.txt
    echo "done $i"
done


for correlation in $(seq 2 4); do
    for i in $(seq $count); do
        echo "SO corre${correlation} pooling ratio 0.2"
        python3 extended-main.py --b 25 --correlation ${correlation} --dataset NCI1 --pooling_ratio 0.2 --read_dataset True --seed $i >> corre${correlation}-NCI1-PR-so.txt
        echo "done $i"
    done
done


for correlation in $(seq 2 4); do
    for i in $(seq $count); do
        echo "MO corre${correlation} pooling ratio 0.2"
        python3 extended-main.py --b 25 --correlation ${correlation} --multi_orbits TRUE --dataset NCI1 --pooling_ratio 0.2 --read_dataset True --seed $i >> corre${correlation}-NCI1-PR-mo.txt
        echo "done $i"
    done
done



####### EXPERIMENTS WITH DIFFERENT MODEL 

for i in $(seq $count); do
    echo "Original M1"
    python3 main.py --dataset NCI1 >> original-NCI1-M1.txt
    echo "done $i"
done



for correlation in $(seq 2 4); do
    for i in $(seq $count); do
        echo "SO corre${correlation} M1"
        python3 extended-main.py --b 25 --correlation ${correlation} --dataset NCI1 --model 1 --read_dataset True --seed $i >> corre${correlation}-NCI1-M1-so.txt
        echo "done $i"
    done
done


for correlation in $(seq 2 4); do
    for i in $(seq $count); do
        echo "MO corre${correlation} M1"
        python3 extended-main.py --b 25 --correlation ${correlation} --multi_orbits TRUE --dataset NCI1 --model 1 --read_dataset True --seed $i >> corre${correlation}-NCI1-M1-mo.txt
        echo "done $i"
    done
done
