SELECT C.*
FROM CFs AS C,
     Prediction_CFs AS PC,
     Instances AS I
WHERE C.CfId = PC.CfId 
  AND PC.PredictionId = pred.PredictionId
  AND I.InstanceId = pred.InstanceId
  AND (C.f1 = I.f1 OR C.f2 != I.f2)


/* A query looking for CFs that either keep a given feature f1 intact, or change both f1 and f2
    For example, for the Adult Income dataset and for f1=education, f2=workclass the constraint that derive from this query for instance x is
   (y_education = x.education) or (y_workclass != x.workclass), where y_f is the (unknown) value of feature f of the counterfactual
   and x.f is the value of feature f of instance x
*/