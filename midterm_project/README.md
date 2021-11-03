My midterm project is about predicting whether a candidate who sign up for a course will change a job or not. The dataset was taken from this link https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists. 

Notes : 
  The dataset is imbalanced.
  Most features are categorical (Nominal, Ordinal, Binary), some with high cardinality.
  Missing imputation can be a part of your pipeline as well.

Features : 
  enrollee_id : Unique ID for candidate
  city: City code
  city_ development _index : Developement index of the city (scaled)
  gender: Gender of candidate
  relevent_experience: Relevant experience of candidate
  enrolled_university: Type of University course enrolled if any
  education_level: Education level of candidate
  major_discipline :Education major discipline of candidate
  experience: Candidate total experience in years
  company_size: No of employees in current employer's company
  company_type : Type of current employer
  lastnewjob: Difference in years between previous job and current job
  training_hours: training hours completed
  target: 0 – Not looking for job change, 1 – Looking for a job change


