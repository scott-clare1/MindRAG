#!/bin/bash

set -e

host="$1"
port="$2"
shift 2
cmd="$@"

# Check for the availability of 'nc' or 'ncat'
command -v nc > /dev/null 2>&1 || { echo >&2 "Error: 'nc' is not installed. Please install 'netcat' or 'ncat'."; exit 1; }

until nc -z -v -w30 "$host" "$port" > /dev/null 2>&1; do
  echo "Server is not reachable yet. Waiting..."
  sleep 5
done

echo "Server is now reachable. Running the command: $cmd"
exec $cmd
