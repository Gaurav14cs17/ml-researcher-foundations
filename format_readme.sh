#!/bin/bash

# Script to format all README files as blog posts

for file in $(find . -name "README.md" ! -path "./images/*"); do
    # Skip if already has capsule-render header
    if grep -q "capsule-render" "$file"; then
        echo "✓ Already formatted: $file"
        continue
    fi
    
    # Get directory name for title
    dirname=$(dirname "$file")
    title=$(basename "$dirname" | sed 's/-/ /g' | sed 's/_/ /g' | sed 's/\b\(.\)/\u\1/g')
    
    # Skip images directories
    if [[ "$dirname" == *"/images"* ]]; then
        echo "⏭ Skipping images: $file"
        continue
    fi
    
    # Create header
    header='<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text='"$title"'&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

'
    
    # Add header to file
    echo "$header" | cat - "$file" > temp && mv temp "$file"
    
    # Add footer if not present
    if ! grep -q "section=footer" "$file"; then
        echo '
---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>' >> "$file"
    fi
    
    echo "✅ Formatted: $file"
done

echo ""
echo "=== Done! ==="
