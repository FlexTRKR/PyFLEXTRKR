#!/bin/bash



# sudo su root -c "sync; echo 3 > /proc/sys/vm/drop_caches"
# ulimit -v 4194304 && 

log_name=$1
log_file="${log_name}.log"


echo "Logging mem usage to $log_file"

index=0  # Initialize the index variable

free -h | awk -v idx="$index" 'BEGIN{OFS="\t"} NR==1{print "Index\t","Type\t" $0} NR==2{print idx, $0}' | tee "$log_file"

while true; do
  # Run the `free` command and append the formatted output to the log file using `tee`
  free -h | awk -v idx="$index" 'BEGIN{OFS="\t"} NR==2{print idx, $0}' | tee -a "$log_file"

  # Increment the index
  ((index++))

  # Sleep for a desired interval before running the loop again
  sleep 1
done