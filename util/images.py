# necessary imports
import numpy as np
import matplotlib.pyplot as plt
from feature_cheese import fCHEESE_extractor as fCH_extractor
from img_util import readImageFile

"""
Helper python script that saves the images related 
to plotting image-mask pairs used in the report.

This includes:
Figure 1, Figure 2, Figure 3 (temp)
"""
# PLOT 1: BLACK MASK
#------------------------------------------------------------
img_rgb, _ = readImageFile("../data/lesion_imgs/PAT_104_1755_320.png")
img_mask = readImageFile("../data/lesion_masks/PAT_104_1755_320_mask.png", is_mask= True)

fig = plt.figure(figsize= (12, 5))
fig1 = fig.add_subplot(1, 2, 1)
fig1.set_axis_off()
fig1.set_title("PAT_104_1755_320.png")
fig1.imshow(img_rgb)

fig2 = fig.add_subplot(1, 2, 2)
fig2.set_axis_off()
fig2.set_title("PAT_104_1755_320_mask.png")
fig2.imshow(img_mask, cmap= 'gray')

fig.savefig("../data/black_mask_example.pdf", dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved as [black_mask_example.pdf] in the /data directory.")
#------------------------------------------------------------

# PLOT 2: CC EXAMPLE
#------------------------------------------------------------
img_rgb, _ = readImageFile("../data/lesion_imgs/PAT_113_172_610.png")
img_mask = readImageFile("../data/lesion_masks/PAT_113_172_610_mask.png", is_mask= True)
img_mask = (img_mask > 127).astype(np.uint8)

mean_cc, raw_cc = fCH_extractor(img_mask, test= True)

fig = plt.figure(figsize= (12, 5))
fig1 = fig.add_subplot(1, 2, 1)
fig1.set_axis_off()
fig1.set_title(f"PAT_113_172_610.png")
fig1.imshow(img_rgb)

fig2 = fig.add_subplot(1, 2, 2)
fig2.set_axis_off()
fig2.set_title(f"Mean cc: {mean_cc} | Raw cc: {raw_cc}")
fig2.imshow(img_mask, cmap= 'gray')

fig.savefig("../data/cc_comparison.pdf", dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved as [cc_comparison.pdf] in the /data directory.")
#------------------------------------------------------------

# PLOT 3: GOOD EXAMPLE OF IMAGE-MASK PAIR
#------------------------------------------------------------
img_rgb, _ = readImageFile("../data/lesion_imgs/PAT_1216_759_542.png")
img_mask = readImageFile("../data/lesion_masks/PAT_1216_759_542_mask.png", is_mask= True)

img_mask = (img_mask > 127).astype(np.uint8)

img_lesion = img_rgb.copy()
img_lesion[img_mask == 0] = [0, 0, 0]

fig = plt.figure(figsize= (12, 5))
fig1 = fig.add_subplot(1, 3, 1)
fig1.set_axis_off()
fig1.set_title(f"PAT_1216_759_542.png")
fig1.imshow(img_rgb)

fig2 = fig.add_subplot(1, 3, 2)
fig2.set_axis_off()
fig2.set_title(f"PAT_1216_759_542_mask.png")
fig2.imshow(img_mask, cmap= 'gray')

fig3 = fig.add_subplot(1, 3, 3)
fig3.set_axis_off()
fig3.set_title(f"PAT_1216_759_542.png - Masked")
fig3.imshow(img_lesion)

fig.savefig("../data/good_example.pdf", dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved as [good_example.pdf] in the /data directory.")
#------------------------------------------------------------