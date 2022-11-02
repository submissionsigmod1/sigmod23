SELECT C.*
FROM CFs AS C,
     Prediction_CFs AS PC,
     Instances AS I
WHERE C.CfId = PC.CfId 
  AND PC.PredictionId = pred.PredictionId
  AND I.InstanceId = pred.InstanceId
  AND (C.F1 != I.F1 OR
       C.F2 != I.F2 OR
       C.F3 != I.F3)

/* A query looking for CFs that change at least one feature in a given set.
   For example, for the Adult Income dataset and a set of features={education, workclass, occupation} the constraint that derive from this query for instance x is
   (y_education != x.education) or (y_workclass != x.workclass) or (y_occupation != x.occupation), where y_f is the (unknown) value of feature f of the counterfactual
   and x.f is the value of feature f of instance x
   */
