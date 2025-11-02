"""
Test furniture detection with the example image
"""
import cv2
import numpy as np

# Load the image
img = cv2.imread('Images/Examples/example.png', 0)

# Find contours
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print(f"Found {len(contours)} contours")

# Filter contours by size to find furniture-like objects
tables = []
beds = []
chairs = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area = w * h
    aspect_ratio = float(w) / h if h > 0 else 0

    # Table: medium size, roughly square or rectangular
    if 500 < area < 5000 and 0.5 < aspect_ratio < 2.0:
        tables.append((x, y, w, h))
        print(f"Possible table: area={area}, ratio={aspect_ratio:.2f}, pos=({x},{y}), size=({w}x{h})")

    # Bed: larger, more rectangular
    elif 2000 < area < 15000 and 1.2 < aspect_ratio < 3.0:
        beds.append((x, y, w, h))
        print(f"Possible bed: area={area}, ratio={aspect_ratio:.2f}, pos=({x},{y}), size=({w}x{h})")

    # Chair: small, roughly square
    elif 50 < area < 800 and 0.7 < aspect_ratio < 1.5:
        chairs.append((x, y, w, h))
        print(f"Possible chair: area={area}, ratio={aspect_ratio:.2f}, pos=({x},{y}), size=({w}x{h})")

print(f"\nDetected:")
print(f"Tables: {len(tables)}")
print(f"Beds: {len(beds)}")
print(f"Chairs: {len(chairs)}")
