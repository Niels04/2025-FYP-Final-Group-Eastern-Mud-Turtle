library(tseries)
library(ggplot2)
library(rprojroot)

# Set working directory to script location
set_script_wd <- function() {
  if (!interactive()) {
    # If run via Rscript or source()
    this_file <- normalizePath(sys.frame(1)$ofile)
  } else if (requireNamespace("rstudioapi", quietly = TRUE) &&
             rstudioapi::isAvailable()) {
    # If running interactively in RStudio
    this_file <- rstudioapi::getSourceEditorContext()$path
  } else {
    stop("Cannot determine script path. Please set working directory manually.")
  }
  
  setwd(dirname(this_file))
  message("Working directory set to: ", getwd())
}

set_script_wd()

#hypothesis testing for method comparison

#recall performance sample for base method:
base <- c(0.5, 0.25, 0.25, 0.3, 0.1875, 0.2, 0.14285714, 0.375, 0.14285714, 0.54545455, 0.14285714, 0.57142857, 0.5, 0.29411765, 0.25, 0.2, 0.28571429, 0.625, 0.5, 0.5)
#recall performance sample for extended method:
extended <- c(0.2, 0.71428571, 0.3, 0.4, 0.33333333, 0.28571429, 0.77777778, 0.33333333, 0.4, 0.76470588, 0.66666667, 0.84615385, 0.46666667, 0.45454545, 0.625, 0.5, 0.7, 0.63157895, 0.6, 0.53846154)

#Jarque-Bera test to test normality of the samples
jb_test_base <- jarque.bera.test(base)
jb_test_extended <- jarque.bera.test(extended)

#print results
cat("Jarque-Bera Test for base model:\n")
print(jb_test_base)
cat("\nJarque-Bera Test for extended model:\n")
print(jb_test_extended)

#QQ-Plot function
plot_qq <- function(dataVec, title) {
  qqnorm(dataVec, main = title)
  qqline(dataVec, col = "red")
}

output_path <- "../result/otherPlots/qqplot_recall.svg"
# Ensure directory exists
dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
# Open SVG device
svg(filename = output_path, width = 8, height = 4)
# Tight layout: remove margins
par(mar = c(2, 2, 2, 2))   # Bottom, left, top, right
par(oma = c(0, 0, 0, 0))   # Outer margins

#make QQ-Plots
par(mfrow = c(1, 2))# Side-by-side plots
plot_qq(base, "QQ-Plot: Base Model Recall")
plot_qq(extended, "QQ-Plot: Extended Model Recall")
par(mfrow = c(1, 1))#Reset layout
dev.off()
#by visual inspection of the QQ-Plots and the high p-values for the Jarque-Bera test,
#we assume normality of the data

#-> can use a t-test (for independent samples because no fixed random state was used to obtain
#the samples)

#check for equality of variances:
var.test(base, extended)

#variances seem to be equal (p-value about 0.5)
#-> do a t-test with pooled variance
t.test(extended, base, var.equal = TRUE, alternative="greater")

#Conclusion: p-value = 0.0007491 and thus we conclude that extended performs better
#at the 99.9% confidence level
getwd()