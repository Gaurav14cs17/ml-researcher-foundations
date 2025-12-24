#!/bin/bash

# Fix broken image paths in README files
echo "Fixing image paths..."

find . -name "README.md" | while read readme; do
    dir=$(dirname "$readme")
    
    # Find all image references in this file
    grep -oE '\./images/[^)"]+\.(png|jpg|svg|gif)' "$readme" 2>/dev/null | while read img_ref; do
        # Check if image exists at referenced path (relative to readme dir)
        full_path="$dir/$img_ref"
        if [[ ! -f "$full_path" ]]; then
            # Image doesn't exist, find it
            img_name=$(basename "$img_ref")
            
            # Search for the image
            found=$(find . -name "$img_name" -type f 2>/dev/null | head -1)
            
            if [[ -n "$found" ]]; then
                # Calculate relative path from readme dir to found image
                # For simplicity, check if image is in local images folder
                local_img="$dir/images/$img_name"
                if [[ -f "$local_img" ]]; then
                    # Image is in local images folder, path should work
                    :
                else
                    echo "README: $readme"
                    echo "  Missing: $img_ref"
                    echo "  Found at: $found"
                fi
            else
                echo "README: $readme"
                echo "  Missing: $img_ref (NOT FOUND ANYWHERE)"
            fi
        fi
    done
done
