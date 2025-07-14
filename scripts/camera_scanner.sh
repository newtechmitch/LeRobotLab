#!/bin/bash

# Camera Resolution Scanner
# Lists video devices and gets their default format info

echo "Listing video devices..."
echo "========================"

# Get the device list
device_output=$(ffmpeg -f avfoundation -list_devices true -i "" 2>&1)

# Extract video devices (everything before "AVFoundation audio devices:")
video_devices=$(echo "$device_output" | awk '/AVFoundation video devices:/,/AVFoundation audio devices:/ {if (!/AVFoundation audio devices:/) print}' | grep -E "\[[0-9]+\]")

echo "$video_devices"
echo ""

# For each video device, get format info
echo "Getting format information for each camera..."
echo "============================================="

echo "$video_devices" | while IFS= read -r line; do
    if [[ $line =~ \[([0-9]+)\] ]]; then
        device_id="${BASH_REMATCH[1]}"
        device_name=$(echo "$line" | sed 's/.*\] //')
        
        echo ""
        echo "Camera [$device_id]: $device_name"
        echo "-----------------------------------"
        
        # Skip screen capture devices as they behave differently
        if [[ "$device_name" == *"Capture screen"* ]]; then
            echo "Skipping screen capture device"
            continue
        fi
        
        # Get format info using the exact command you specified
        ffmpeg -f avfoundation -i "$device_id" -t 1 -f null - 2>&1 | grep -E "(Input|Stream|Video:|fps|format)"
    fi
done

echo ""
echo "Done."