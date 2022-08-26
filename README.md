# A Comparative Study for Time-to-Event Analysis and Survival Prediction for Heart Failure Condition using Machine Learning Techniques

This work is published as part of the [Journal of Electronics, Electromedical Engineering, and Medical Informatics](https://jeeemi.org/index.php/jeeemi) and can be accessed online at the [Journal Page](https://doi.org/10.35882/jeeemi.v3i3.2). Please [cite](#citeAs) the work if you find these codes useful for your work. 

This work is an open-access and licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa] [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa] 

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

**<ins>LINK TO ABSTRACT</ins>:** http://jeeemi.org/index.php/jeeemi/article/view/225 </br>
**<ins>LINK TO FULL PDF</ins>:** http://jeeemi.org/index.php/jeeemi/article/view/225/94

## ABSTRACT

Heart Failure, an ailment in which the heart isn’t functioning as effectively as it should, causing in an insufficient cardiac output. The effectual functioning of the human body is dependent on how well the heart is able to pump oxygenated, and nutrient rich blood to the tissues and cells. Heart failure falls into the category of cardiovascular diseases - the disorders of the heart and blood vessels. One of the leading causes of global deaths resulting in an estimated 17.9 million deaths globally every year. The condition of heart failure results out of structural changes to the cardiac muscles majorly in the left ventricle. The weakened muscles cause the ventricle to lose its ability to contract completely. Since the left ventricle generates the required pressure for blood circulation, any kind of a failure condition results in the reduction of cardiac power output. This study aims to conduct a thorough survival analysis and survival prediction on the data of 299 patients classified into the class III/IV of heart failure and diagnosed with left ventricular systolic dysfunction. Survival analysis involves the study of the effect of a mediation assessed by measuring the number of subjects survived after that mediation over a period of time. The time starting from a distinct point to the occurrence of a certain event, for example death is known as survival time and the corresponding analysis is known as survival analysis. The analysis was performed using the methods of Kaplan-Meier (KM) estimates and Cox Potential Hazard regression. KM plots showed the survival estimates as a function of each clinical feature and how each feature at various levels affect survival over the period of time. Cox regression modelled the hazard of death event around the clinical features used for the study. As a result of the analysis, ejection fraction, serum creatinine, time and age were identified as highly significant and major risk factors in the advanced stages of heart failure. Age and rise in level of serum creatinine have a deleterious effect on the survival chances. Ejection Fraction has a beneficial effect on survival and with a unit increase in the in the EF level the probability of death event decreases by ~5.2%. Higher rate of mortality is observed during the initial days post diagnosis and the hazard gradually decreases if patients have lived for a certain number of days. Hypertension and anemic condition also seem to be high risk factors. Machine learning classification models for survival prediction were built using the most significant variables found from survival analysis. SVM, decision tree, random forest, XGBoost, and LightGBM algorithm were implemented, and all the models seem to perform well enough. However, the availability of more data will make the models more stable and robust. Smart solutions, like this can reduce the risk of heart failure condition by providing accurate prognosis, survival projections, and risk predictions. Technology and data can combine together to address any disparities in treatment, design better care plan, and improve patient health outcomes. Smart health AI solutions would enhance healthcare policies, enable physicians to look beyond the conventional practices, and increase the patient satisfaction levels not only in case of heart failure conditions but healthcare in general.

 ## DATASET
 
 The dataset contains cardiovascular medical records taken from 299 patients. The patient cohort comprised of 105 women and 194 men between 40 and 95 years in age. All patients in the cohort were diagnosed with the systolic dysfunction of the left ventricle and had previous history of heart failures. As a result of their previous history every patient was classified into either class III or class IV of New York Heart Association (NYHA) classification for various stages of heart failure.
 
 #### FEATURE DESCRIPTION
 
 Feature | Explanation | Measurement	| Range
------------- | ------------- |------------- | -------------
Age	|Age of the patient	|Years |	[40,..., 95]
Anaemia	|Decrease of red <br> blood cells or hemoglobin |	Boolean|	0, 1
High blood pressure |	If a patient has hypertension |	Boolean	 |0, 1
Creatinine phosphokinase<br>(CPK) |	Level of the CPK enzyme <br>in the blood |	mcg/L|	[23,..., 7861]
Diabetes|	If the patient has diabetes |	Boolean	| 0, 1
Ejection fraction|	Percentage of blood leaving<br>the heart at each contraction|Percentage	|	[14,..., 80]
Sex	| Woman or man |	Binary|	0, 1
Platelets|	Platelets in the blood|	kiloplatelets/mL|	[25.01,..., 850.00]
Serum creatinine|	Level of creatinine in the blood|	mg/dL|	[0.50,..., 9.40]
Serum sodium|	Level of sodium in the blood|	mEq/L|	[114,..., 148]
Smoking|	If the patient smokes|	Boolean	|0, 1
Time|	Follow-up period|	Days|	[4,...,285]
DEATH EVENT<br>(TARGET)|	If the patient died during the follow-up period|	Boolean|	0, 1

**NOTE: mcg/L: micrograms per liter. mL: microliter. mEq/L: milliequivalents per litre**

#### SOURCE
 Ahmad, Tanvir; Munir, Assia; Bhatti, Sajjad Haider; Aftab, Muhammad; Ali Raza, Muhammad (2017): DATA_MINIMAL.. PLOS ONE. Dataset-   https://doi.org/10.1371/journal.pone.0181001.s001 </br>
 Link: https://plos.figshare.com/articles/dataset/Survival_analysis_of_heart_failure_patients_A_case_study/5227684/1
 
 ## PAPERS WITH CODE

Link: [https://paperswithcode.com/paper/malaria-parasite-detection-using-efficient](https://paperswithcode.com/paper/a-comparative-study-for-time-to-event)

 <h2 id="citeAs">HOW TO CITE</h2>
 If you find this work helpful for your study, please cite the paper as follows - </br></br>
 
Mishra, S. (2022) “A Comparative Study for Time-to-Event Analysis and Survival Prediction for Heart Failure Condition using Machine Learning Techniques”, Journal of Electronics, Electromedical Engineering, and Medical Informatics, 4(3), pp. 115-134. doi: 10.35882/jeeemi.v4i3.225.
