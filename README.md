# PFE-alzheimer-disease-classification

According to the World Health Organization, Alzheimer's disease is now among the top ten causes of death worldwide. Today, 116 years after the discovery of this disease, we still have no way to prevent it, stop it or even slow it down. Traditionally, the diagnosis of Alzheimer's disease has relied primarily on clinical observation and cognitive assessment. But recently, with the advancement of technology and its integration into medicine, medical imaging has become a way to diagnose, monitor, and treat many diseases because of its speed, accuracy, cost, and minimal risk to the health of the patient. One of the areas affected by this progression is the diagnosis of Alzheimer's disease using magnetic resonance imaging.

## Files description

* `Application` : the developed UI tool, refer to [this](#how-to-run-the-application) for more info.

* `Preprocessing` : all files used to preprocess the data.

* `AD_classification.ipynb` : all the tests that were done to train the model and choose the best hyperparameter.

## Description of the approach
### Data
The Alzheimer’s Disease Neuroimaging Initiative [ADNI](https://adni.loni.usc.edu/) unites researchers with study data as they work to define the progression of Alzheimer’s disease (AD). ADNI researchers collect, validate and utilize data, including MRI and PET images, genetics, cognitive tests, CSF and blood biomarkers as predictors of the disease. Study resources and data from the North American ADNI study are available through this website, including Alzheimer’s disease patients, mild cognitive impairment subjects, and elderly controls.
<br/>
in our work, we used the ADNI1 dataset that consisted of T1 weighted MRI, images were classified into 3 classes:
* AD : Alzheimer’s disease patients
* MCI : Mild cognitive impairment subjects
* NC : Elderly controls

### Preprocessing
we applied a simple preprocing techinque that consisted of :

1. **Affine Transformation** : The main objective of this pre-processing is to limit the variations in positioning, orientation, shape and size of the images of the images in our study, we use the [Fsl FLIRT](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT) tool to linearly register all scans to the T1 [MNI 152 Template](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases) with 2mm Isotropy.

<p align="center">
  <img src="1.png" />
</p>

1. **Skull Striping** : is the process of eliminating non-brain tissues and any unnecessary information for that we applied the included mask with the MN 152 template.

<p align="center">
  <img src="2.png" />
</p>

1. **Construction of the 2D image** : instead of extracting slices randomly and assuming that they contain the most relevant information, we instead extracted random slices based on entropy, because the most informative slices will maximize the entropy value. then we selected the 16 most informative slices. These slices are then placed in a 4 by 4 matrix.

<p align="center">
  <img src="3.png" />
</p>

1. **Histogram Equalization** : To improve the contrast of the images used, we apply histogram equalization.

<p align="center">
  <img src="4.png" />
</p>

### Classifier
we built a model based on the resnet50: 
<p align="center">
  <img src="5.png" />
</p>

<p align="center"><b>
A summary table of the model parameters.</b>
</p>

 **Parameters**            | **Value**                          | **Reason**                                                                                      
---------------------------|------------------------------------|-------------------------------------------------------------------------------------------------
 **input shape**           | 436×364×3                          | is the size of the image to be classified                                                       
 **Optimization function** | Adam                               | The most used optimizer in the literature                                                       
 **batch size**            | 4                                  | After several trials, we have found that small batch sizes give better results                  
 **epochs**                | 50                                 | The maximum number we were able to reach while doing our test, without google colab blocking us 
 **output shape**          | Probability vector with 3 elements | /                                                                                               

## How to run the Application
in order to be able to run the application you need to :

1. Install [Fsl](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation) following the instruction here.
2. Run :
    ```bash
    pip install requirements.txt
    ```
3. Unzip the `row.part01.rar` in `\Application\Models`

## Some pictures of the application
<p align="center">
  <img src="1.gif" />
  <img src="2.gif" />
  <img src="3.gif" />
  <img src="4.gif" />
</p>

## Conclusion
Overall, we can draw the following conclusions:

* Deep learning methods and more precisely neural networks are able to give a reliable diagnosis of Alzheimer's disease based on MRI.
* The ADNI1 database contains enough quality images to build good models. However, the processing of this information is a bit more complicated especially for the extraction of non-brain tissue.

Following our approach, we had good results, a validation accuracy of 93.40%, an average precision of 92.02%, and an average recall of 93.68%, All while avoiding data leakage.

We see as a follow-up to this work several perspectives such as:
* The use of sagittal or coronal planes instead of axial planes.
* The application of other data augmentation techniques such as image blurring and noise injection.

* Use of an adaptive learning rate that can be dynamically adjusted during model training.

* Other possible improvements would be to use higher resolution images (3T ADNI scans).


