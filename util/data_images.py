"""Helper python script that creates and saves in the /data directory 
all plots used in the Data subsection of the report.

"""

from img_util import ImageDataLoader as IDL
from img_util import readImageFile
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from scipy.ndimage import binary_dilation, generate_binary_structure

# example of totally black mask
# -----------------------------------------
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

fig.savefig("../data/black_mask_example.png", dpi=300)
plt.close()
print("Plot saved as [black_mask_example.png] in the /data directory.")
# -----------------------------------------

# example of cc results with and without dilation
# -----------------------------------------

def fCH_extractorTest(mask) -> int:
    structure = generate_binary_structure(mask.ndim, 1)
    cc_list = []

    for i in range(5, 26, 5):
        dilated_mask = binary_dilation(mask, structure= structure, iterations= i)
        labeled_mask = label(dilated_mask, connectivity= 2)
        cc = labeled_mask.max()
        cc_list.append(cc)

    # also normal labelling to plot
    raw_cc = label(mask, connectivity= 2).max()

    return sum(cc_list) // len(cc_list), raw_cc

img_rgb, _ = readImageFile("../data/lesion_imgs/PAT_113_172_610.png")
img_mask = readImageFile("../data/lesion_masks/PAT_113_172_610_mask.png", is_mask= True)
img_mask = (img_mask > 127).astype(np.uint8)

mean_cc, raw_cc = fCH_extractorTest(img_mask)

fig = plt.figure(figsize= (12, 5))
fig1 = fig.add_subplot(1, 2, 1)
fig1.set_axis_off()
fig1.set_title(f"PAT_113_172_610.png")
fig1.imshow(img_rgb)

fig2 = fig.add_subplot(1, 2, 2)
fig2.set_axis_off()
fig2.set_title(f"Mean cc: {mean_cc} | Raw cc: {raw_cc}")
fig2.imshow(img_mask, cmap= 'gray')

fig.savefig("../data/cc_comparison.png", dpi=300)
plt.close()
print("Plot saved as [cc_comparison.png] in the /data directory.")
# -----------------------------------------

# good example of image - mask pair
# -----------------------------------------
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

fig.savefig("../data/good_example.png", dpi=300)
plt.close()
print("Plot saved as [good_example.png] in the /data directory.")
# -----------------------------------------