from PIL import Image
import os

def overlay(file1, file2, output, alpha=0.5):
    img1 = Image.open(file1).convert("RGBA")
    img2 = Image.open(file2).convert("RGBA")
    img2 = img2.resize(img1.size)
    blended = Image.blend(img1, img2, alpha=alpha)
    blended.save(output)

overlay("pricemovement_noobs.png", "pricemovement.png", "pmOverlaid.png")
overlay("orders_noobs.png", "orders.png", "ordersOverlaid.png")
overlay("profitability_noobs.png", "profitability.png", "profitabilityOverlaid.png")