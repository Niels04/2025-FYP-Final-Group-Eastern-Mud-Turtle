# necessary imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
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

# PLOT 3: OPTIMAL rate_hair() PARAMETERS
#------------------------------------------------------------
"""Helper python script to find optimal parameters for the rate_hair function,
based on computed hair ratios with and without image blurs on 300 images.
The images assigned to group C (mandatory assignment) are used as a test set.
The accuracy will be printed when running the file.
Will plot and save the results in the data directory"""

# compute optimal thresholds
print("> rate_hair() threshold test results:")
ah = pd.read_csv("../data/annotations.csv")

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
plt.savefig("../data/3D_accuracy_thresholds.pdf", dpi=300)
plt.close()
print("Plot saved as [3D_accuracy_thresholds.pdf] in the /data directory.")


# accuracy on original annotations

test_set = ah[ah['Group_ID'] == 'C']

true_labels = test_set['Rating_Final']
ratios = test_set['Hair_Ratio']
predicted_labels = np.array([0 if r < 0.020 else 1 if r < 0.118 else 2 for r in ratios])
print(f"Accuracy of rate_hair() on test data (annotations from group C mandatory assigment): {accuracy_score(true_labels, predicted_labels):.4f}")