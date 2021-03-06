# alphabet_optimization

## **Overview**
* The goal of this analysis was to create an algorithm for the Alphabet Soup foundation to try to predict whether applicants to their program will be successful with their funding.
--------------------------------------------------------------
## **Initial Data Preprocessing**
The data contained information about 34,299 applications received by Alphabet Soup. In addition to 2 identification columns (EIN & NAME), there were 10 other factors included in the dataset (APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT, & IS_SUCCESSFUL). 

* Seemingly irrelevant info (& thus removed early in preprocessing): EIN & NAME

* Model target: IS_SUCCESSFUL (1=yes, 0=no)

* Model features (options per category): APPLICATION_TYPE (17), AFFILIATION (6), CLASSIFICATION(71), USE_CASE(5), ORGANIZATION(4), STATUS(2), INCOME_AMT(9), SPECIAL_CONSIDERATIONS(2), ASK_AMT(8747)

### Data transformations:
The volume of options in both APPLICATION_TYPE & CLASSIFICATION required grouping some of the less populated categories with each other. Without having additional information about application types and classification categories, these aggregations were based solely on numbers. Ideally, having more information about what the types & classifications actually mean might allow for more meaningful aggregation--for instance, grouping small categories with larger ones that might be conceptually similar rather than just grouping all of the small categories together. I believe that type of information could lead to better model options.  

* APPLICATION_TYPE was reduced from 17 categories to 9--8 categories that each had >500 applications & an 'Other' category to capture the remaining 276 applications.
* CLASSIFICATION was reduced from 71 categories to 6--5 categories that each had >1800 application & an 'Other' category to capture the remaining 2261 applications.
* Of the factors, only STATUS & ASK_AMT were numeric, thus all other factors were one-hot encoded via .get_dummies, resulting in a dataframe with 44 columns. 
* SPECIAL_CONSIDERATIONS split into 2 columns SPECIAL_CONSIDERATIONS_Y & SPECIAL_CONSIDERATIONS_N, because Y/N is binary in nature, keeping both columns was redundant & thus SPECIAL_CONSIDERATIONS_N was dropped from the dataframe. 

Data was then split & scaled for the modeling process. 
* IS_SUCCESSFUL was saved as the y-variable for the model & test and removed from the larger array of prediction factors. 
* The default proportions for the train_test_split method were used. 
* After splitting data, the factor arrays were scaled using the StandardScaler() method.

The training array had 25724 rows of data with 42 columns.

--------------------------------------------
## **Compiling, Training, and Evaluating the Model**

### *_Initial Model_* (did not reach 75% threshold)
* initial model used nodes, layers, & activation functions consistent with in-class activities
* specifically: 
  *  sequential model with 3 dense layers 
  *  hidden layers using 'relu' activation
  *  output layer using 'sigmoid' activation. 
  * ![initial_model](https://user-images.githubusercontent.com/83370545/137649489-8cb8c742-7d58-4c9a-a618-daf77f22d651.png)
* additionally:
  * compiled with: (loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]) based on in-class activities 
  * callbacks recorded every 5 iterations 
  * model ran for 100 epochs
* RESULTS:
  * model leveled off relatively early with accuracy ~ 0.74 & loss ~0.54. 
  * ![initial_model_fit](https://user-images.githubusercontent.com/83370545/137651381-78c14bd3-fb34-468c-97bb-222a798939cf.png)
  * performance on the testing dataset slightly worse (accuracy ~ 0.73 & loss ~ 0.56)
  * ![initial_model_test](https://user-images.githubusercontent.com/83370545/137651410-0fcdab89-4c3c-49cb-9de3-7bd539b69808.png)
 
Though, close, the initial model did not exceed the desired threshold of 75% accuracy. Attempts to optimize the model will now be discussed.

-----------------
## Optimization Attempts

### *_Optimization Attempt 1: Added neuron layer_* (did not improve the model)

* additional dense neuron layer with 20 units & relu activation added to the previous model
* ![optimize_1_model](https://user-images.githubusercontent.com/83370545/137651555-ee595f9e-cd19-4215-8743-f2ea872173bf.png)
* other parameters and methods same as initial trial

* RESULTS:
  * like previous model 
    * accuracy hovered around 0.74
    * loss was between 0.53 and 0.54
    * ![optimize_1_fit](https://user-images.githubusercontent.com/83370545/137651629-926b3fa8-d5bd-43e7-824c-d242faf06b5e.png)
 * Performance on the testing dataset were almost identical to the initial model
    * ![optimize_1_test](https://user-images.githubusercontent.com/83370545/137651661-8dbd25ca-cfa9-47cc-b792-91d65acc7d0d.png)


### *_Additional Data Exploration_*
Factors that had not been changed in the original preprocessing were investigated to see if creating additional bins might be useful. 

AFFILIATION: 
* largely either Independent (18480) or CompanySponsored (15705)
* remaining categories very sparse: 
  * Family/Parent (64)
  * National (33)
  * Regional (13)
  * Other (4) 
* Lumping all of the remaining categories together was considered
  * but rejected because conceptually Family/Parent group seemed different than the others, particularly when looking at ask_amt
  * ![affiliation_ask](https://user-images.githubusercontent.com/83370545/137651823-72de3333-1cb7-42d4-a9d7-af04edeeb124.png)
  * may be an option to consider for future models
 

USE_CASE: 
* was heavily Preservation (28095) & ProductDev (5671)
* remaining categories were much smaller: 
  * CommunityServ (384)
  * Heathcare (146)
  * Other (3)
* commuity service & health care seem conceptually similar, BUT comparisons between ASK_AMT of these groups suggested they might not be 
  * ![usecase_ask](https://user-images.githubusercontent.com/83370545/137652192-123833cb-bd42-42be-95bc-dd6888e965f0.png)


ORGANIZATION:
* mostly Trust (23515) & Association (10255)
* smaller groups: Co-operative (486), & Corporation (43)
  * due to conceptual differences between co-operative and corporation merging into a single group not considered
* seems like a case where knowing more about the definitions of what these labels mean might have allowed adding the smaller groups to either of the larger ones 
* ![oraganization_ask](https://user-images.githubusercontent.com/83370545/137652302-767c5be5-d54b-4280-bae5-4642671f01f1.png)


INCOME_AMT:
* 9 groups
  * largest group income=0 (24388)
  * 1-9999 (728)
  * 10000-24999 (543)
  * 25000-99999 (3747)
  * 100000-499999 (3374)
  * 1M-5M (955)
  * 5M-10M (185)
  * 10M-50M (240)
  * 50M+ (139) 
* clear case where combining smaller groups together _could_ lead to conceptually meaningless categories if the actual meaning of the categories is not considered 
* ![income_ask](https://user-images.githubusercontent.com/83370545/137652548-2d2c7d10-83ef-4404-acd0-d6a22b6a8ee5.png)

* RECODED for _some_ of the final models
  * categories replaced with a numeric reverse-rank system 
    * wherein the 0 income category = 0, 1-9999=1, ... 50M+=8
    * lower numbers in the recoding indicated lower levels of income
    * higher numbers in the recoding indicated higher levels of income
  * recoding in this way is conceptually sound
    * categories are representative of actual numeric values
    * reduced the number of columns that resulted from one-hot encoding  

ASK_AMT: 
* for some models: 
  * applications with requests over $100M were removed from the dataset
  * resulted in 101 applications being dropped from the dataset 
  * 0.29% of original data & thus acceptable amount of outliers to be excluded



### *_Optimization Attempt 2: remove applications asking >$100M; other parameters same as initial model_* (slight improvement from previous models)
* ![optimize_2_model](https://user-images.githubusercontent.com/83370545/137652793-7d1e7ef5-a675-4986-a5ae-050225ffa3b1.png)
* outperformed the previous models slightly
  * fit accuracy of 0.744 & loss of 0.531 
  * ![optimize_2_fit](https://user-images.githubusercontent.com/83370545/137652829-85ee2ef6-59fc-47fa-9e32-d8bf9107a125.png)
* performance on the testing data similar to previous models
  * ![optimize_2_test](https://user-images.githubusercontent.com/83370545/137652865-03698799-7018-4b26-a290-dd59b442ca7d.png)


### *_Optimization Attempt 3: same as Optimization 2 + removing STATUS & SPECIAL_CONSIDERATIONS_* (similar to initial model)
* Due to minimal variation, STATUS & SPECIAL_CONSIDERATION removed from dataset
  *  ![status](https://user-images.githubusercontent.com/83370545/137652986-cf1c90d6-2aed-4047-b7fc-11df63d6bbb8.png)
  *  ![spec_con](https://user-images.githubusercontent.com/83370545/137652997-6f335400-cc43-46ad-82e2-282342d2f0aa.png)
* resulted in 40 columns for the training array.
  * ![optimize_3_model](https://user-images.githubusercontent.com/83370545/137653019-e6419a36-195c-49de-aef6-e2a903173673.png)
* model fit very similar to first 2 models, hovering around 0.74 accuracy & 0.53 loss.
  * ![optimize_3_fit](https://user-images.githubusercontent.com/83370545/137653062-85d385c2-0ba3-4893-8317-ceb9a8e4b318.png)
* testing results similar to previous models
  * ![optimize_3_test](https://user-images.githubusercontent.com/83370545/137653090-e8a513e8-7957-4a79-b3b1-3c191c0cbfaf.png)


### *_Optimization Attempt 4_* (fit accuracy > 0.75)

* Preprocessing:
  - kept EIN column
    - hunch related to prior assignments & the instructions
    - I know that conceptually, I shouldn't, but other assignments had ID numbers as easter eggs of sorts
  - Dropped: NAME, STATUS, SPECIAL_CONSIDERATIONS
  - INCOME_AMT converted to the reverse-rank system discussed earlier
  - APPLICATION_TYPE & CLASSIFICATION processed in same way as all other runs
  - ASK_AMT all applications were included
  - 33 columns in training set

* Model:
  - increased units on 2nd hidden layer
  - added 3rd hidden layer
    - ![optimize_4_model](https://user-images.githubusercontent.com/83370545/137653336-b0819759-7d4d-4953-a45f-cf75624b57b4.png)
  - all other options related to compiling & training matched previous models

* Results:
  - 0.7500 accuracy on Epoch 62/100 
    - ![optimize_4_epoch62](https://user-images.githubusercontent.com/83370545/137653384-28bbffe5-bb8e-4d9b-96e2-3c7aa91f391f.png)
  - oscillated for awhile
  - final accuracy 0.753 & loss 0.51 
    - ![optimize_4_fit](https://user-images.githubusercontent.com/83370545/137653412-8b05fbc4-7473-4662-9ed5-83818f9d2f5a.png)
  - testing slightly better 
    - ![optimize_4_test](https://user-images.githubusercontent.com/83370545/137653430-c8629523-1c72-4389-8dce-f15437804ec4.png)


### *_Optimization Attempt 5 (fit accuracy 0.755); identical to Optimization 4, but removed >$100M_*

Removing applications above $100M made slight improvement over previous model with regard to fit accuracy, but slightly worse test accuracy.

* Results:
  - 0.7505 accuracy on Epoch 55/100 
    - ![optimize_5_epoch55](https://user-images.githubusercontent.com/83370545/137653471-ee849d6b-3da8-4310-87bf-ddf34a1ddeb4.png)
  - oscillated for awhile
  - final accuracy 0.755 & loss 0.506 
    - ![optimize_5_fit](https://user-images.githubusercontent.com/83370545/137653495-e88d497b-91d3-4b80-a822-e65232faac30.png)
  - testing similar to other models
    - ![optimize_5_test](https://user-images.githubusercontent.com/83370545/137653561-207e2f1a-a6f2-4e28-b49b-f32e0cec4560.png)

-----------------------------------

## **Summary** 

In general, cracking the 75% threshhold was not easy. If I were to run more models (which I may just to see what happens), I would start with the data & parameters used in either Optimization 4 or 5 (only difference is whether >$100M is in or out of the set) and remove the EIN column. Adding the additional neuron layer was also a confound that I should not have done at the same time as switching out some of the variables, but I was getting a bit impatient and just wanted to try some stuff. So with that in mind, there's no way to know whether the additional layer or the variable changes were more important to the model change. 

Being able to ensure that conglomerate groups are conceptually meaningful/related might also improve the model. Good data can make a big difference in statistical analyses.

Because this is a classification problem, other methods that might be considered include: logistic regression, k-nearest neighbors, decision tree classification, and random forest classification. Of these, logistic regression & k-nearest neighbors tend to be the easier to explain, so I'd probably attempt one of those first. 
