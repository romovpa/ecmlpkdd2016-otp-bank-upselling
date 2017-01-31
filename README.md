# Solution of ECML/PKDD 2016 Discovery Challenge: Credit Card Upselling Task

The competition task was to predict whether the user may apply for a credit card in the next 6 months, which is important for OTP bank to iniate an aupselling. Organizers provided one year of historic data on user attributes, activity events and geolocation. It was organized as a classification contest, evaluated with Area Under the Curve (AUC).

- [Full description of the competition](https://dms.sztaki.hu/ecml-pkkd-2016/#/app/tasks)
- [Slides for ECML/PKDD workshop](http://www.slideshare.net/romovpa/a-simple-yet-efficient-method-for-a-credit-card-upselling-prediction)
- [Paper](https://github.com/romovpa/ecmlpkdd2016-otp-bank-upselling/raw/master/paper/romov-discovery-challenge-2016-paper.pdf)

## Features

To represent user with a feature vector, several logical group of features were designed. They described below. Some features could not been computed for some users due to lack of user information or events of some kind, in this case the feature has missing value.

### Personal

Categorical features from client profile:
- gender
- age category (35 and less, 36--65, 65 and more)
- location category (capital, city, village)
- income category (no income, low, medium, high)

The features encoded two ways: 1) with one-hot encoding (to eliminate order of categories) 2) with one integer number (to take the order into account).

### Cards and Wealth

All those features are computed on the first six months of the year:
- Number of months when user had a card, the same aggregation for being wealthy
- Last month of the period when user had a card / was wealthy
- Number of indicator changes from having a card / being wealthy to a month without this indicator, and vice versa

### Event counters

First feature of this group --- total number of events. Then for each categorical variable describing an event two vector of counters are computed: the first represent exact number of events having specific value of categorical faatures, the second represent ratio of events having specific value in all events. 

Event counter features were calculated for the following categorical features:
- Type of activity (point of sale, webshop, branch)
- Time rounded to three ranges (05-11h, 12-18h, 19-04h)
- Event location category (capital, city or village)
- Anonymized market category groups (7 unique categories)
- Type of card used (credit or debit)
- Amount of money spent in three categories (low, medium, high)
- Weekday

### Number of unique shops

Total number of unique places of events and the number by channel type: number of unique branches of the bank, web shops and point of sale.

### Client activity

We call the client active in the specific period when he committed at least one transaction in the period. As for features describing client activity in the train period, the following features were computed:

- Number of days / weeks the client was active
- Duration in days from the first event to the last
- Average number of active days / weeks
- Average number of inactive days / weeks
- Active days rate

### Geolocation features

We add coordinates of user home address as known client features.

The geographical coordinates of branches of the bank and points of sale were provided. To represent geographical statistics for events of each user, the idea was to compute distances and angles of event points to the client home and the capital city (Budapest) aggregated with several statistics: average, minimum, maximum, standard deviation, 20/50/80-quantiles

## Prediction model

To build classification model, the model of choice was [XGBoost](https://xgboost.readthedocs.io/) with binary classification objective and default parameter settings: 
- maximum tree depth = 3
- 100 iterations  
- learning rate = 0.1
We tried to tune maximum tree depth, learning rate and different chemes of filling missing values, but default settings were sufficient for our score. 

Tuning parameters such as maximum tree depth, learning rate and several schemes of filling missing values didn't provide any significant gain on AUC-ROC estimated with simple stratified cross-validation on the one year data. Moreover, fair cross-validation wasn't feasible due to lack of year-to-year overlaps. Hence the idea was to stick with the Occam's Razor and build a simple yet reasonable model to counter potential overfitting.


## How to reproduce the results

1. Download and extract the data into `data` folder
2. Run feature extraction: `python upselling_features.py`. This could take several hours.
3. Run `python upselling_model.py` to generate submission file and model dump (in `result` folder).

Python scripts depends on several packages listed in `requirements.txt`.

