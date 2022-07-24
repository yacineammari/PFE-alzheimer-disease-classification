# PFE-alzheimer-disease-classification

According to the World Health Organization, Alzheimer's disease is now among the top ten causes of death worldwide. Today, 116 years after the discovery of this disease, we still have no way to prevent it, stop it or even slow it down. Traditionally, the diagnosis of Alzheimer's disease has relied primarily on clinical observation and cognitive assessment. But recently, with the advancement of technology and its integration into medicine, medical imaging has become a way to diagnose, monitor, and treat many diseases because of its speed, accuracy, cost, and minimal risk to the health of the patient. One of the areas affected by this progression is the diagnosis of Alzheimer's disease using magnetic resonance imaging.

## Files description

* `Application` : the develped UI tool reffer to this for more info.

* `Preprocessing` : all files used to preprced the data.

* `AD_classification.ipynb` : all the test effected to train the model

## Description of the approche
* talk about the data
* talk about the aproche preproccing
* talk about classifyer and it paramter

## How to run the Application
Before the use of applcation 

1. Run 
    ```bash
    pip install requirements.txt
    ```
2. Installed Fsl fooliwn the instraction here

3. Unzip the `row.part01.rar` in the path Application\Models

inset images and gif of the application 

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


