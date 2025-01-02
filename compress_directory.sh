#!/bin/bash

# Check if input directory provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_directory>"
    exit 1
fi

INPUT_DIR="$1"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory $INPUT_DIR does not exist"
    exit 1
fi

# Check if identifiers.txt exists
if [ ! -f "identifiers.txt" ]; then
    echo "Error: identifiers.txt not found in current directory"
    exit 1
fi

# Process identifiers.txt and find completed identifiers
echo "Finding completed identifiers..."
while IFS=',' read -r identifier status || [ -n "$identifier" ]; do
    if [ "$status" = "Done" ]; then
        echo "Processing completed identifier: $identifier"
        
        # Check if directory exists
        if [ -d "$INPUT_DIR/$identifier" ]; then
            echo "Creating zip for $identifier"
            
            # Create zip file
            (cd "$INPUT_DIR" && zip -r "${identifier}.zip" "$identifier")
            
            # Check if zip was successful
            if [ $? -eq 0 ] && [ -f "$INPUT_DIR/${identifier}.zip" ]; then
                echo "Removing original directory for $identifier"
                rm -rf "$INPUT_DIR/$identifier"
                echo "Completed processing $identifier"
            else
                echo "Error: Failed to create zip for $identifier"
            fi
        else
            echo "Warning: Directory for $identifier not found"
        fi
    fi
done < identifiers.txt

echo "Compression process complete"
