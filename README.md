# Projects in Data Science (2025)

Final Assignment

#### Overview

This is a template repository for the final assignment of course "Projects in Data Science." You should refer to this repository in your final hand-in.

If using github.itu.dk, you need to download the repository and make your own. 

If you are using general Github, you can clone or fork the repository directly. If your usernames do not give sufficient hints as to who you are, you can tell the TAs how to match them. 

Your repository MUST be named 2025-FYP-groupXX where XX is your group name (e.g. groupPandas). 

Look at the slides of the previous two weeks for details of the hand-in. 

#### Python environment

Follow TA instructions when setting up the Python environment before running any code. Remember to export your Python library requirements by `pip freeze > requirements.txt` and attach it to the repo so we can evaluate your scripts.

#### File Hierarchy

The file hierarchy of your hand-in repo should be as follows:

```
2025-FYP/
├── data/               # unzip the dataset and put it here (remove in your hand-in)
│   ├── img_001.jpg
│   ......
│   └── img_XXX.jpg
│ 
├── util/
│   ├── __init__.py
│   ├── img_util.py     # basic image read and write functions
│   ├── inpaint.py      # image inpainting function
│   ├── feature_A.py    # code for feature A extraction
│   ├── feature_B.py    # code for feature B extraction
│   ├── feature_C.py    # code for feature C extraction
│   ......
│   └── classifier.py   # code for training, validating, and testing the classifier
│ 
├── result/
│   ├── result_baseline.csv      # your results on the baseline setup
│   ├── result_extended.csv      # your results on the extended setup
│   └── report.pdf               # your report in PDF
│ 
├── main_demo.py        # demo script (reference setup, remove in your hand-in)
├── main_baseline.py    # complete script (baseline setup)
├── main_extended.py    # complete script (extended setup)
├── dataset.csv            # all image file names, ground-truth labels, and chosen features
└── README.md
```

**Notes:**

1. DO NOT upload your data (images) to Github.
2. When the same code block needs to be executed multiple times in the script, make it a custom function instead. All the custom functions and modules, such as image read and write, should be grouped into different files under the *"util"* subfolder, based on the task they are designed for. Do not put everything in a single Python file or copy-paste the same code block across the script.
