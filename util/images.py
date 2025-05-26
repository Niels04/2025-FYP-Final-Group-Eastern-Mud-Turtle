# necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from feature_cheese import fCHEESE_extractor as fCH_extractor
from img_util import readImageFile
import cv2
from skimage.transform import resize
from skimage.transform import rotate
from sklearn.cluster import KMeans
from pathlib import Path

import feature_B

#setup directories
_FILE_DIR = Path(__file__).resolve().parent#obtain directory of this file
_PROJ_DIR = _FILE_DIR.parent#obtain main project directory
_DATA_DIR = _PROJ_DIR / "data"#obtain data directory
_RESULT_DIR = _PROJ_DIR / "result"#obtain results directory
_PLT_DIR = _RESULT_DIR / "otherPlots"
_IMG_DIR = _PROJ_DIR / "data/lesion_imgs"
_MASK_DIR = _PROJ_DIR / "data/lesion_masks"

"""
Helper python script that saves the images related 
to plotting image-mask pairs used in the report.

This includes:
Figure 1, Figure 2, Figure 3 (temp)
"""

# PLOT 1: BLACK MASK
#------------------------------------------------------------
img_rgb, _ = readImageFile(str(_IMG_DIR / "PAT_104_1755_320.png"))
img_mask = readImageFile(str(_MASK_DIR / "PAT_104_1755_320_mask.png"), is_mask= True)

fig = plt.figure(figsize= (12, 5))
fig1 = fig.add_subplot(1, 2, 1)
fig1.set_axis_off()
fig1.set_title("PAT_104_1755_320.png")
fig1.imshow(img_rgb)

fig2 = fig.add_subplot(1, 2, 2)
fig2.set_axis_off()
fig2.set_title("PAT_104_1755_320_mask.png")
fig2.imshow(img_mask, cmap= 'gray')

fig.savefig(str(_PLT_DIR / "black_mask_example.pdf"), dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved as [black_mask_example.pdf].")
#------------------------------------------------------------

# PLOT 2: CC EXAMPLE
#------------------------------------------------------------
img_rgb, _ = readImageFile(str(_IMG_DIR / "PAT_113_172_610.png"))
img_mask = readImageFile(str(_MASK_DIR / "PAT_113_172_610_mask.png"), is_mask= True)
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

fig.savefig(str(_PLT_DIR / "cc_comparison.pdf"), dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved as [cc_comparison.pdf].")
#------------------------------------------------------------

# PLOT 3: GOOD EXAMPLE OF IMAGE-MASK PAIR
#------------------------------------------------------------
img_rgb, _ = readImageFile(str(_IMG_DIR / "PAT_1216_759_542.png"))
img_mask = readImageFile(str(_MASK_DIR / "PAT_1216_759_542_mask.png"), is_mask= True)

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

fig.savefig(str(_PLT_DIR / "good_example.pdf"), dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved as [good_example.pdf].")
#------------------------------------------------------------

# PLOT 3: OPTIMAL rate_hair() PARAMETERS
#------------------------------------------------------------
"""Helper python script to find optimal parameters for the rate_hair function,
based on computed hair ratios with and without image blurs on 300 images.
The images assigned to group C (mandatory assignment) are used as a test set.
The accuracy will be printed when running the file.
Will plot and save the results in the data directory"""

# compute optimal thresholds
print("> rate_hair() threshold test results:")
ah = pd.read_csv(str(_DATA_DIR / "annotations.csv"))

# get information from the train set
train_set = ah[ah['Group_ID'] != 'C']
true_labels = train_set['Rating_Final']
ratios = train_set['Hair_Ratio']

# initialize variables to store best results
best_acc = 0
best_t1, best_t2 = None, None

# list to save values to plot later
acc_lB = []
t1_lB = []
t2_lB = []

# iterate through possible threshold values
for t1 in np.linspace(0.0, 1.0, 100):
    for t2 in np.linspace(t1 + 0.01, 1.0, 100):

        t1_lB.append(t1)
        t2_lB.append(t2)

        predicted_labels = np.array([0 if r < t1 else 1 if r < t2 else 2 for r in ratios])

        acc = accuracy_score(true_labels, predicted_labels)
        acc_lB.append(acc)
        
        if acc > best_acc:
            best_acc = acc
            best_t1, best_t2 = t1, t2

print(f"WITH BLUR | Best thresholds: t1 = {best_t1:.3f}, t2 = {best_t2:.3f}; with accuracy = {best_acc:.4f}")


# compute optimal thresholds (no blur)

# get information from the train set
train_set = ah[ah['Group_ID'] != 'C']
true_labels = train_set['Rating_Final']
ratios = train_set['Hair_Ratio_noB']

# initialize variables to store best results
best_acc = 0
best_t1, best_t2 = None, None

# list to save values to plot later
acc_l = []
t1_l = []
t2_l = []

# iterate through possible threshold values
for t1 in np.linspace(0.0, 1.0, 100):
    for t2 in np.linspace(t1 + 0.01, 1.0, 100):

        t1_l.append(t1)
        t2_l.append(t2)

        predicted_labels = np.array([0 if r < t1 else 1 if r < t2 else 2 for r in ratios])

        acc = accuracy_score(true_labels, predicted_labels)
        acc_l.append(acc)
        
        if acc > best_acc:
            best_acc = acc
            best_t1, best_t2 = t1, t2

print(f"  NO BLUR | Best thresholds: t1 = {best_t1:.3f}, t2 = {best_t2:.3f}; with accuracy = {best_acc:.4f}")


# plot results

fig = plt.figure(figsize=(12, 5))

# plot for normal images
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(t1_l, t2_l, acc_l, c= acc_l, cmap= 'viridis', alpha= 0.6)

ax1.view_init(elev=30, azim=135)

ax1.set_title("Accuracy score\n(without image blur)")
ax1.set_xlabel('Lower threshold')
ax1.set_ylabel('Upper threshold')
ax1.set_zlabel('Accuracy')

# plot for blurred images
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(t1_lB, t2_lB, acc_lB, c= acc_lB, cmap= 'viridis', alpha= 0.6)

ax2.view_init(elev=30, azim=135)

ax2.set_title("Accuracy score\n(with image blur)")
ax2.set_xlabel('Lower threshold')
ax2.set_ylabel('Upper threshold')
ax2.set_zlabel('Accuracy')

plt.tight_layout()
plt.savefig(str(_PLT_DIR / "3D_accuracy_thresholds.pdf"), dpi=300)
plt.close()
print("Plot saved as [3D_accuracy_thresholds.pdf].")


# accuracy on original annotations

test_set = ah[ah['Group_ID'] == 'C']

true_labels = test_set['Rating_Final']
ratios = test_set['Hair_Ratio']
predicted_labels = np.array([0 if r < 0.020 else 1 if r < 0.118 else 2 for r in ratios])
print(f"Accuracy of rate_hair() on test data (annotations from group C mandatory assigment): {accuracy_score(true_labels, predicted_labels):.4f}")


# IMAGE 4: Visualization of Feature B for Open Question
#-------------------------------------------------------

def draw_sector_overlay(image, center, nSectors, line_color='lightgreen'):
    """
    Overlay sector boundary lines and labels on the image.

    :param image: 2D (grayscale) or 3D (color) image as NumPy array.
    :param center: (x, y) tuple of the center point.
    :param radius: Length of the radial lines (in pixels).
    :param sector_angles_deg: List of (start_deg, end_deg) for each sector.
    :param line_color: Color of the overlay lines (default: light green).
    """
    rad = 100

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    cx, cy = center

    sectorDeg = 360 / nSectors
    for i in range(nSectors):
        #calculate ending and starting degree
        theta_start = (np.pi/2) - np.deg2rad(i*sectorDeg)
        theta_end = (np.pi/2) - np.deg2rad((i+1)*sectorDeg)

        # Draw boundary lines
        for theta in [theta_start, theta_end]:#basically just go over the two items in the list
            x_end = cx + rad * np.cos(theta)
            y_end = cy + rad * np.sin(theta)
            ax.plot([cx, x_end], [cy, y_end], color=line_color, linewidth=1.5)

        # Place label in the center of the sector
        mid_theta = (theta_start + theta_end) / 2
        x_label = cx + (rad * 0.2) * np.cos(mid_theta)
        y_label = cy + (rad * 0.2) * np.sin(mid_theta)
        ax.text(x_label, y_label, str(i), color='lightgreen', fontsize=12, ha='center', va='center')

    ax.set_axis_off()
    plt.tight_layout()

def fB_formula_visualization(img, mask, nSectors=8):
    """Extract the \"irregular Boder\" feature,
    which is a number from 0 to 1 that is a measure
    for the difference between the center intensity
    and border intensity of the lesion.
    
    :param img: the image to process
    :param mask: mask to apply to the image
    :return: border irregularity measure from
    0(regular) to 1(irregular)"""
    #Preprocess: Apply a gaussean blur
    img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0)

    cutImg = feature_B.cut_im_by_mask(img, mask)
    cutImgGray = cv2.cvtColor(cutImg, cv2.COLOR_RGB2GRAY)#convert image to grayscale for gradient analysis
    cutMask = feature_B.cut_mask(mask)
    mX, mY = feature_B.find_midpoint_v4(cutMask)

    #store max gradient values for the sectors
    gradScores = []

    sectorDeg = int(np.ceil(360 / nSectors))
    for deg in range(0, 360, sectorDeg):
        #analyze gradient in this sector
        avgMaxGrad = feature_B.analyze_sector_gradients(cutImgGray, cutMask, (mX, mY), deg, deg+sectorDeg)

        gradScores.append(avgMaxGrad)

    draw_sector_overlay(cutImg, (mX, mY), nSectors)
    plt.imshow(cutMask, cmap="Reds", alpha=0.1)
    plt.axis("off")
    plt.savefig(str(_RESULT_DIR / "otherPlots/featureB_formula.png"), dpi=300, bbox_inches="tight")
    gradScores = np.array(gradScores)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(gradScores.reshape(-1, 1))
    # Get cluster centers (means)
    centers = kmeans.cluster_centers_.flatten()  # Shape (2,)
    # Identify the cluster with the lower mean
    lower_mean_cluster = np.argmax(centers)
    # Get labels
    labels = kmeans.labels_
    # Count sectors assigned to the lower-mean cluster
    count = np.sum(labels == lower_mean_cluster)
    return count


imName = "PAT_615_1167_722.png"
maskName = imName.split(".")[0] + "_mask" + ".png"
testImg = cv2.imread(str(_IMG_DIR /  imName))
testImg = cv2.cvtColor(testImg, cv2.COLOR_BGR2RGB)
testMask = cv2.imread(str(_MASK_DIR / maskName), cv2.IMREAD_GRAYSCALE)
_, testMask = cv2.threshold(testMask, 127, 255, cv2.THRESH_BINARY)

fB_formula_visualization(testImg, testMask)