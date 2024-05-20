#!/bin/sh

# rm -rf original-PROTEINS.txt
# rm -rf corre2-PROTEINS-so.txt
# rm -rf corre3-PROTEINS-so.txt
# rm -rf corre4-PROTEINS-so.txt
# rm -rf corre2-PROTEINS-mo.txt
# rm -rf corre3-PROTEINS-mo.txt
rm -rf corre4-PROTEINS-mo.txt


rm -rf original-PROTEINS-PR.txt
rm -rf corre2-PROTEINS-PR-so.txt
rm -rf corre3-PROTEINS-PR-so.txt
rm -rf corre4-PROTEINS-PR-so.txt
rm -rf corre2-PROTEINS-PR-mo.txt
rm -rf corre3-PROTEINS-PR-mo.txt
rm -rf corre4-PROTEINS-PR-mo.txt

rm -rf original-PROTEINS-M1.txt
rm -rf corre2-PROTEINS-M1-so.txt
rm -rf corre3-PROTEINS-M1-so.txt
rm -rf corre4-PROTEINS-M1-so.txt
rm -rf corre2-PROTEINS-M1-mo.txt
rm -rf corre3-PROTEINS-M1-mo.txt
rm -rf corre4-PROTEINS-M1-mo.txt


count=10

for i in $(seq $count); do
    echo "Original"
    python3 main.py --b 59 --dataset PROTEINS >> original-PROTEINS.txt
    echo "done $i"
done

for correlation in $(seq 2 4); do
    for i in $(seq $count); do
        echo "SO corre${correlation} pooling ratio 0.2"
        python3 extended-main.py --b 59 --correlation ${correlation} --dataset PROTEINS --read_dataset True --seed $i >> corre${correlation}-PROTEINS-so.txt
        echo "done $i"
    done
done


for correlation in $(seq 2 4); do
    for i in $(seq $count); do
        echo "MO corre${correlation} pooling ratio 0.2"
        python3 extended-main.py --b 59 --correlation ${correlation} --multi_orbits TRUE --dataset PROTEINS --read_dataset True --seed $i >> corre${correlation}-PROTEINS-so.txt
        echo "done $i"
    done
done



########## EXPERIMENTS WITH DIFFERENT pooling ratios ##########:


for i in $(seq $count); do
    echo "Original pooling ratio 0.2"
    python3 main.py --b 59 --dataset PROTEINS --pooling_ratio 0.2 >> original-PROTEINS-PR.txt
    echo "done $i"
done


for correlation in $(seq 2 4); do
    for i in $(seq $count); do
        echo "SO corre${correlation} pooling ratio 0.2"
        python3 extended-main.py --b 59 --correlation ${correlation} --dataset PROTEINS --pooling_ratio 0.2 --read_dataset True --seed $i >> corre${correlation}-PROTEINS-PR-so.txt
        echo "done $i"
    done
done


for correlation in $(seq 2 4); do
    for i in $(seq $count); do
        echo "MO corre${correlation} pooling ratio 0.2"
        python3 extended-main.py --b 59 --correlation ${correlation} --multi_orbits TRUE --dataset PROTEINS --pooling_ratio 0.2 --read_dataset True --seed $i >> corre${correlation}-PROTEINS-PR-mo.txt
        echo "done $i"
    done
done



####### EXPERIMENTS WITH DIFFERENT MODEL 

for i in $(seq $count); do
    python3 main.py --dataset PROTEINS >> original-PROTEINS-M1.txt
    echo "done $i"
done



for correlation in $(seq 2 4); do
    for i in $(seq $count); do
        echo "SO corre${correlation} pooling ratio 0.2"
        python3 extended-main.py --b 59 --correlation ${correlation} --dataset PROTEINS --model 1 --read_dataset True --seed $i >> corre${correlation}-PROTEINS-M1-so.txt
        echo "done $i"
    done
done


for correlation in $(seq 2 4); do
    for i in $(seq $count); do
        echo "MO corre${correlation} pooling ratio 0.2"
        python3 extended-main.py --b 59 --correlation ${correlation} --multi_orbits TRUE --dataset PROTEINS --model 1 --read_dataset True --seed $i >> corre${correlation}-PROTEINS-M1-mo.txt
        echo "done $i"
    done
done

