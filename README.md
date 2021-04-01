# kul_advanalytics
Git project for our Advanced Analytics in Business assignments

# Extra information
This [link](https://nvie.com/posts/a-successful-git-branching-model/) gives some background information on the purpose of Git. 
You will see that this can get quite complex in case you work with an entire development team. However, for this assignment it will not get this complex. 
Regardless, it can give you a nice understanding of the concepts behind version control.


# Features
* Out of 308 fraudulent claims, only 1 involved an injured driver
* Resp. 15% and 25% of all claims caused by theft or fire where fraudulent
* Some policy_holder_expert_id's have almost only fraudulent cases
* Some driver_expert_id's have almost only fraudulent cases
* The higher the claim amount, the higher the probability of fraud
  (for instance, all cases with more than 17k claimed were fraudulent)
  Problem: we don't have claim_amount in the test set. How can we predict claim_amount from the other values?
  
Not significant:
* brand, alcohol, police, policy_holder_country, policy_coverage_type, country, 
* nr of days between accident happend and claim filed
* claim postal code = policy postal code
* sex of the driver

# TODO
* Try featuretools.com for automated feature engineering
* Predict claim amount first (a regression model), then use that in the fraud prediction model  
* Treat policy_holder_expert_id and  driver_expert_id as factor  => now implemented as list of blacklisted experts
* Treat claim_postal_code as factor
* Fraud seems more prevalent in higher claim_vehicle_cyl
* Postal code seems relevant, for instance many fraud cases where claimed in Moeskroen (not surprisingly)
* Try XGBoost
