# alphabet_optimization

**Overview**
The goal of this analysis was to create an algorithm for the Alphabet Soup foundation to try to predict whether applicants to their program will be successful with their funding. 

**Results**

** _Initial Data Preprocessing_**
The data contained information about 34,299 applications received by Alphabet Soup. In addition to 2 identification columns (EIN & NAME), there were 10 other factors included in the dataset (APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, ASK_AMT, & IS_SUCCESSFUL). 

* Seemingly irrelevant info (& thus removed early in preprocessing): EIN & NAME

* Model target: IS_SUCCESSFUL (1=yes, 0=no)

* Model features (options per category): APPLICATION_TYPE (17), AFFILIATION (6), CLASSIFICATION(71), USE_CASE(5), ORGANIZATION(4), STATUS(2), INCOME_AMT(9), SPECIAL_CONSIDERATIONS(2), ASK_AMT(8747)

* Data transformations:
The volume of options in both APPLICATION_TYPE & CLASSIFICATION required grouping some of the less populated categories with each other. Without having additional information about application types and classification categories, these aggregations were based solely on numbers. Ideally, having more information about what the types & classifications actually mean might allow for more meaningful aggregation--for instance, grouping small categories with larger ones that might be conceptually similar rather than just grouping all of the small categories together. I believe that type of information could lead to better model options.  

APPLICATION_TYPE was reduced from 17 categories to 9--8 categories that each had >500 applications & an 'Other' category to capture the remaining 276 applications.

CLASSIFICATION was reduced from 71 categories to 6--5 categories that each had >1800 application & an 'Other' category to capture the remaining 2261 applications.

Of the factors, only STATUS & ASK_AMT were numeric, thus all other factors were one-hot encoded via .get_dummies, resulting in a dataframe with 44 columns. SPECIAL_CONSIDERATIONS split into 2 columns SPECIAL_CONSIDERATIONS_Y & SPECIAL_CONSIDERATIONS_N, because Y/N is binary in nature, keeping both columns was redundant & thus SPECIAL_CONSIDERATIONS_N was dropped from the dataframe. 

Data was then split & scaled for the modeling process. IS_SUCCESSFUL was saved as the y-variable for the model & test and removed from the larger array of prediction factors. The default proportions for the train_test_split method were used. After splitting data, the factor arrays were scaled using the StandardScaler() method.

The training array had 25724 rows of data with 42 columns.

** _Compiling, Training, and Evaluating the Model_**

* How many neurons, layers, and activation functions did you select for your neural network model, and why?
* Initial Model
The initial model used nodes, layers, & activation functions were chosen to be consistent with in-class activities.
Specifically, it was a sequential model with 3 dense layers with hidden layers using 'relu' activation and the output layer using 'sigmoid' activation. [/images/initial_model.png] 

The model was compiled with these parameters: (loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]) based on in-class activities. Callbacks were set to record every 5 iterations and the model ran for 100 epochs. The model leveled off relatively early with accuracy ~ 0.74 & loss ~0.54. [/images/initial_model_fit.png] Performance on the testing dataset were slightly worse (accuracy ~ 0.73 & loss ~ 0.56). [/images/initial_model_test.png]

Though, close, the initial model did not exceed the desired threshold of 75% accuracy. Attempts to optimize the model will now be discussed.







* Were you able to achieve the target model performance?

* What steps did you take to try and increase model performance?

3. **Summary**: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.
