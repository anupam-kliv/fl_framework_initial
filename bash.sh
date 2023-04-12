cd server 
python start_server.py &
sleep 10s
cd ../client 
python client.py &
wait