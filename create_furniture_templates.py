"""
Create furniture template images for detection
"""
from PIL import Image, ImageDraw
import numpy as np

# Create table template (rectangular shape)
def create_table_template():
    # Create a white background
    img = Image.new('L', (100, 100), color=255)
    draw = ImageDraw.Draw(img)

    # Draw a rectangle for table (typical blueprint representation)
    draw.rectangle([20, 30, 80, 70], outline=0, width=3)

    img.save('Images/Models/Furniture/table.png')
    print("Created: Images/Models/Furniture/table.png")

# Create bed template (rectangle with pillow indicator)
def create_bed_template():
    # Create a white background
    img = Image.new('L', (100, 120), color=255)
    draw = ImageDraw.Draw(img)

    # Draw bed frame (rectangle)
    draw.rectangle([15, 15, 85, 105], outline=0, width=3)

    # Draw pillow area (line at top)
    draw.line([15, 30, 85, 30], fill=0, width=2)

    img.save('Images/Models/Furniture/bed.png')
    print("Created: Images/Models/Furniture/bed.png")

# Create chair template (small square/rectangle)
def create_chair_template():
    # Create a white background
    img = Image.new('L', (60, 60), color=255)
    draw = ImageDraw.Draw(img)

    # Draw a small rectangle for chair
    draw.rectangle([15, 15, 45, 45], outline=0, width=2)

    img.save('Images/Models/Furniture/chair.png')
    print("Created: Images/Models/Furniture/chair.png")

if __name__ == "__main__":
    create_table_template()
    create_bed_template()
    create_chair_template()
    print("\nAll furniture templates created successfully!")
