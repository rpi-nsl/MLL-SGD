#!/bin/bash

data=( 1 2 0 )
model=( 1 3 0 )

for i in "${!data[@]}"; do
    # Exp 1
    python mll_sgd.py --data ${data[i]} --model ${model[i]} --hubs 1 --workers 100 --tau 1 --q 32 --graph 5 --epochs 200 --batch 300 --fed True  
    python mll_sgd.py --data ${data[i]} --model ${model[i]} --hubs 1 --workers 100 --tau 32 --q 1 --graph 5 --epochs 200 --batch 300 --fed True 
    python mll_sgd.py --data ${data[i]} --model ${model[i]} --hubs 10 --workers 10 --tau 8 --q 4 --graph 5 --epochs 200 --batch 300 --fed True 
    python mll_sgd.py --data ${data[i]} --model ${model[i]} --hubs 10 --workers 10 --tau 4 --q 8 --graph 5 --epochs 200 --batch 300  --fed True 

    # Baselines
    python mll_sgd.py --data ${data[i]} --model ${model[i]} --hubs 1 --workers 100 --tau 1 --q 32 --graph 5 --epochs 200 --batch 300  

    # Probability skews  
    python mll_sgd.py --data ${data[i]} --model ${model[i]} --hubs 10 --workers 10 --tau 8 --q 4 --graph 5 --epochs 200 --batch 300 
    python mll_sgd.py --data ${data[i]} --model ${model[i]} --hubs 10 --workers 10 --tau 8 --q 4 --graph 5 --epochs 200 --batch 300 --prob 1 
    python mll_sgd.py --data ${data[i]} --model ${model[i]} --hubs 10 --workers 10 --tau 8 --q 4 --graph 5 --epochs 200 --batch 300 --prob 2 
    python mll_sgd.py --data ${data[i]} --model ${model[i]} --hubs 10 --workers 10 --tau 8 --q 4 --graph 5 --epochs 200 --batch 300 --prob 3 
    python mll_sgd.py --data ${data[i]} --model ${model[i]} --hubs 10 --workers 10 --tau 8 --q 4 --graph 5 --epochs 200 --batch 300 --prob 4 

    # Hub size differences
    python mll_sgd.py --data  --model 3 --hubs 10 --workers 10 --tau 32 --q 1 --graph 6 --epochs 200 --batch 300 
    python mll_sgd.py --data  --model 3 --hubs 10 --workers 10 --tau 8 --q 4 --graph 6 --epochs 200 --batch 300 
    python mll_sgd.py --data  --model 3 --hubs 5 --workers 20 --tau 8 --q 4 --graph 6 --epochs 200 --batch 300 
    python mll_sgd.py --data  --model 3 --hubs 20 --workers 5 --tau 8 --q 4 --graph 6 --epochs 200 --batch 300 

    # Probability slots exps
    python mll_sgd.py --data  --model 3 --hubs 10 --workers 10 --tau 32 --q 1 --graph 5 --epochs 200 --batch 300 
    python mll_sgd.py --data  --model 3 --hubs 10 --workers 10 --tau 32 --q 1 --graph 5 --epochs 200 --batch 300 --prob 5 
    python mll_sgd.py --data  --model 3 --hubs 10 --workers 10 --tau 8 --q 4 --graph 5 --epochs 200 --batch 300 --prob 5 
done
