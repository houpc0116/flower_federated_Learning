#!/bin/bash

echo "Starting server"
python server.py --model='cnn' --fr_rate=1.0 --datasets='mnist' --strategy='fedavg' &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 0 4`; do
    echo "Starting client $i"
        python client_FedBN.py --client=5 --model='cnn' --datasets='mnist' --strategy='fedavg' --mode='fedbn' --local_ep='1' --partition=${i} &    
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
