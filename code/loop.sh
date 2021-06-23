while true
do
	echo "$(date +%T) Pulling Data..."
	sh pulling_data.sh

	echo "$(date +%T) Training Model..."
	python3 train.py
	
	for i in $(seq 1 7)
	do 
		echo "$(date +"20%y-%m-%d %T") Sleeping for $((7-i+1)) hours more..."
		sleep 1s
	done
done
