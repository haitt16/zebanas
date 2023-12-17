input_file="test.txt"

idx=1
while read -r line;
do
#    echo "$line" ;
    python scripts/block_latency.py "${line}" $idx
    idx=$(($idx+1))
done < $input_file