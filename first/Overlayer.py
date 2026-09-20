from PIL import Image
import os

def overlay(file1, file2, output, alpha=0.5):
    img1 = Image.open(file1).convert("RGBA")
    img2 = Image.open(file2).convert("RGBA")
    img2 = img2.resize(img1.size)
    blended = Image.blend(img1, img2, alpha=alpha)
    blended.save(output)

cancel_dir = "Cancel_env"
base_dir = "env"

overlay(
    os.path.join(cancel_dir, "pricemovement_c.png"),
    os.path.join(base_dir, "pricemovement.png"),
    "pmOverlaid.png"
)
overlay(
    os.path.join(cancel_dir, "orders_c.png"),
    os.path.join(base_dir, "orders.png"),
    "ordersOverlaid.png"
)
overlay(
    os.path.join(cancel_dir, "profitability_c.png"),
    os.path.join(base_dir, "profitability.png"),
    "profitabilityOverlaid.png"
)