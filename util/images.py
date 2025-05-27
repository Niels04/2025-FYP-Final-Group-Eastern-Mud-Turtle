# necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.cluster import KMeans
from pathlib import Path
from sklearn.metrics import accuracy_score
from feature_cheese import fCHEESE_extractor as fCH_extractor
from img_util import readImageFile, rate_hair
from feature_A import fA_formula
import feature_B
from feature_B import fB_formula
from feature_C import fC_formula




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
print("Plot saved as [black_mask_example.pdf] in the result/otherPlots folder.")
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
print("Plot saved as [cc_comparison.pdf] in the result/otherPlots folder.")
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
print("Plot saved as [good_example.pdf] in the result/otherPlots folder.")
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

ax1.set_zlim(0, 1)

# plot for blurred images
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(t1_lB, t2_lB, acc_lB, c= acc_lB, cmap= 'viridis', alpha= 0.6)

ax2.view_init(elev=30, azim=135)

ax2.set_title("Accuracy score\n(with image blur)")
ax2.set_xlabel('Lower threshold')
ax2.set_ylabel('Upper threshold')
ax2.set_zlabel('Accuracy')

ax2.set_zlim(0, 1)

plt.tight_layout()
plt.savefig(str(_PLT_DIR / "3D_accuracy_thresholds.pdf"), dpi=300)
plt.close()
print("Plot saved as [3D_accuracy_thresholds.pdf] in the result/otherPlots folder.")


# accuracy on original annotations

test_set = ah[ah['Group_ID'] == 'C']

true_labels = test_set['Rating_Final']
ratios = test_set['Hair_Ratio']
predicted_labels = np.array([0 if r < 0.020 else 1 if r < 0.118 else 2 for r in ratios])
print(f"Accuracy of rate_hair() on test data (annotations from group C mandatory assigment): {accuracy_score(true_labels, predicted_labels):.4f}")
#-------------------------------------------------------

# PLOT 4: Visualization of Feature B for Open Question
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
    plt.savefig(str(_PLT_DIR / "featureB_formula.png"), dpi=300, bbox_inches="tight")
    print(f"Plot saved as [featureB_formula.png] in the result/otherPlots folder.")
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
#-------------------------------------------------------

# PLOT 5: ABC FORMULA FEATURES ACCURACY TABLE
#-------------------------------------------------------
# visual annotations (true labels)
imgs = {
    "PAT_56_86_479.png"     : {'A' : 2, 'B' : 5, 'C' : 2},
    "PAT_59_46_537.png"     : {'A' : 2, 'B' : 5, 'C' : 2},
    "PAT_70_107_591.png"    : {'A' : 2, 'B' : 0, 'C' : 3},
    "PAT_109_868_113.png"   : {'A' : 2, 'B' : 0, 'C' : 2},
    "PAT_320_681_410.png"   : {'A' : 2, 'B' : 0, 'C' : 2},
    "PAT_324_1465_43.png"   : {'A' : 2, 'B' : 0, 'C' : 2},
    "PAT_340_714_68.png"    : {'A' : 2, 'B' : 6, 'C' : 3},
    "PAT_471_909_394.png"   : {'A' : 2, 'B' : 4, 'C' : 3},
    "PAT_490_933_17.png"    : {'A' : 1, 'B' : 6, 'C' : 1},
    "PAT_627_1188_503.png"  : {'A' : 2, 'B' : 0, 'C' : 3},
    "PAT_656_1246_483.png"  : {'A' : 2, 'B' : 4, 'C' : 2},
    "PAT_680_1289_585.png"  : {'A' : 2, 'B' : 4, 'C' : 2},
    "PAT_754_1429_380.png"  : {'A' : 2, 'B' : 5, 'C' : 3},
    "PAT_795_1508_925.png"  : {'A' : 2, 'B' : 0, 'C' : 1},
    "PAT_884_1683_538.png"  : {'A' : 2, 'B' : 2, 'C' : 4},
    "PAT_895_1699_872.png"  : {'A' : 2, 'B' : 2, 'C' : 3},
    "PAT_966_1825_584.png"  : {'A' : 2, 'B' : 2, 'C' : 3},
    "PAT_995_1867_165.png"  : {'A' : 2, 'B' : 0, 'C' : 3},
    "PAT_1113_458_387.png"  : {'A' : 1, 'B' : 6, 'C' : 2},
    "PAT_1259_892_793.png"  : {'A' : 2, 'B' : 3, 'C' : 4},
    "PAT_1286_1000_517.png" : {'A' : 1, 'B' : 8, 'C' : 2},
    "PAT_1420_1460_951.png" : {'A' : 2, 'B' : 6, 'C' : 3},
    "PAT_1653_2916_346.png" : {'A' : 2, 'B' : 2, 'C' : 3},
    "PAT_1698_3122_83.png"  : {'A' : 1, 'B' : 8, 'C' : 1},
    "PAT_1928_3876_437.png" : {'A' : 2, 'B' : 8, 'C' : 3},
    "PAT_2017_4164_500.png" : {'A' : 2, 'B' : 0, 'C' : 4},
    "PAT_2103_4581_72.png"  : {'A' : 2, 'B' : 8, 'C' : 2}
}

# set up directories and open csv
metadata_dir = "../data/metadata.csv"
img_dir = "../data/lesion_imgs/"
mask_dir ="../data/lesion_masks/"
md = pd.read_csv(metadata_dir)

# prepare arrays to store predictions
A_preds = [None] * len(imgs)
B_preds = [None] * len(imgs)
C_preds = [None] * len(imgs)

# iterate through images
i = 0
for img in imgs.keys():

    # get the full path of the image
    img_path = os.path.join(img_dir, img)

    # get the name of the corresponding mask
    name, ext = os.path.splitext(img)
    mask_name = f"{name}_mask{ext}"
    # get the full path of the mask
    mask_path = os.path.join(mask_dir, mask_name)

    # load image and mask
    img_rgb, img_gray = readImageFile(img_path)
    mask_gs = readImageFile(mask_path, is_mask= True)
    mask = (mask_gs > 127).astype(np.uint8) # mask as binary
    # extract features
    A_val = fA_formula(mask)
    B_val = fB_formula(img_rgb, mask)
    C_val = fC_formula(img_rgb, mask)
    
    # append predictions
    A_preds[i] = A_val
    B_preds[i] = B_val
    C_preds[i] = C_val

    i += 1

# compute accuracy for each feature:
A_values = [v['A'] for v in imgs.values()]
B_values = [v['B'] for v in imgs.values()]
C_values = [v['C'] for v in imgs.values()]

A_accuracy = accuracy_score(A_values, A_preds)
B_accuracy = accuracy_score(B_values, B_preds)
C_accuracy = accuracy_score(C_values, C_preds)

# print results
print("Accuracy of Formula features A, B, C\non 27 manually annotated images:")
print(f"Prediction accuracy for A feature: {A_accuracy}")
print(f"Prediction accuracy for B feature: {B_accuracy}")
print(f"Prediction accuracy for C feature: {C_accuracy}")


# create dataframe
data = {
    ("Feature A", "predicted") : A_preds,
    ("Feature A", "actual")    : A_values,
    ("Feature B", "predicted") : B_preds,
    ("Feature B", "actual")    : B_values,
    ("Feature C", "predicted") : C_preds,
    ("Feature C", "actual")    : C_values,
}
index = imgs.keys()
df = pd.DataFrame(data, index=index)

# latex_text = df.to_latex()
# with open("../data/formula_table.tex", "w") as f:
#     f.write(latex_text)

# print(f"Latex code for the table saved to [../data/formula_table.tex]")

# plot table
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# format column headers
columns_flat = ["{} {}".format(*col) for col in df.columns]
cell_text = df.values.tolist()

# draw table
table = plt.table(cellText=cell_text,
                  rowLabels=df.index,
                  colLabels=columns_flat,
                  cellLoc='center',
                  loc='center')

# apply color formatting
for i, row in enumerate(df.index):
    for j, (feat, typ) in enumerate(df.columns):
        if typ == "predicted":
            actual_val = df.iloc[i][(feat, "actual")]
            pred_val = df.iloc[i][(feat, "predicted")]
            cell = table[i+1, j]  # +1 because row 0 is header
            if pred_val == actual_val:
                cell.set_facecolor("#d0f0c0")  # light green
            elif abs(pred_val - actual_val) == 1:
                cell.set_facecolor("#fff7b2") # light yellow
            elif abs(pred_val - actual_val) == 2:
                cell.set_facecolor("#ffd59a") # light orange
            else:
                cell.set_facecolor("#f4a6a6")  # light red

plt.tight_layout()
plt.savefig(str(_PLT_DIR / "formula_table.pdf"), dpi=300, bbox_inches='tight')
print(f"Formula features Table saved as [formula_table.pdf] in the /result/otherPlots folder.")
#-------------------------------------------------------

# PLOT 6: GOOD EXAMPLE OF rate_hair() 
#-------------------------------------------------------
img_rgb, _ = readImageFile(str(_IMG_DIR / "PAT_1483_1678_538.png"))

ratio, label, mask = rate_hair(img_rgb)
fig = plt.figure(figsize= (10, 5))
fig1 = fig.add_subplot(1, 2, 1)
fig1.set_axis_off()
fig1.set_title(f"PAT_1483_1678_538.png")
fig1.imshow(img_rgb)

fig2 = fig.add_subplot(1, 2, 2)
fig2.set_axis_off()
fig2.set_title(f"Hair mask - Predicted label: {label}")
fig2.imshow(mask, cmap= 'gray')
fig.savefig(str(_PLT_DIR / "good_hair.pdf"), dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved as [good_hair.pdf] in the result/otherPlots folder.")
#-------------------------------------------------------

# PLOT 7: BAD EXAMPLE OF rate_hair() 
#-------------------------------------------------------
img_rgb, _ = readImageFile(str(_IMG_DIR / "PAT_115_1138_970.png"))

ratio, label, mask = rate_hair(img_rgb)
fig = plt.figure(figsize= (10, 5))
fig1 = fig.add_subplot(1, 2, 1)
fig1.set_axis_off()
fig1.set_title(f"PAT_115_1138_970.png")
fig1.imshow(img_rgb)

fig2 = fig.add_subplot(1, 2, 2)
fig2.set_axis_off()
fig2.set_title(f"Hair mask - Predicted label: {label}")
fig2.imshow(mask, cmap= 'gray')
fig.savefig(str(_PLT_DIR / "bad_hair.pdf"), dpi=300, bbox_inches='tight')
plt.close()
print("Plot saved as [bad_hair.pdf] in the result/otherPlots folder.")
#-------------------------------------------------------