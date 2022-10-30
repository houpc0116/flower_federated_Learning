#!/bin/bash

echo "Starting server"
python server.py --model='resnet' --fr_rate=1.0 --strategy='fedadagrad' --gpu='4' &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 0 19`; do
    echo "Starting client #$i"
    if [ $i -lt 5 ]; then
        python client.py --client=5 --model='resnet' --strategy='fedadagrad' --partition=${i} --gpu='4' &
    elif [ $i -lt 10 ]; then
        python client.py --client=5 --model='resnet' --strategy='fedadagrad' --partition=${i} --gpu='5' &
    elif [ $i -lt 15 ]; then
        python client.py --client=5 --model='resnet' --strategy='fedadagrad' --partition=${i} --gpu='6' &
    else 
        python client.py --client=5 --model='resnet' --strategy='fedadagrad' --partition=${i} --gpu='7' &
    fi    
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
