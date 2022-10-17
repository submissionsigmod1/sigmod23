SELECT C.*
FROM CFs AS RC,
     Prediction_CFs AS PC,
     Instances AS I
WHERE C.CfId = PC.CfId 
  AND PC.PredictionId = pred.PredictionId
  AND I.InstanceId = pred.InstanceId
  AND C.F1 != I.F1
  AND C.F2 IN (SELECT DISTINCT F2
              FROM Instances
              ORDER BY F2 
              LIMIT K)


/* A query looking for CFs that change feature f1 and change feature f2 to one of 𝑘 smallest values of feature f2 in the dataset.
    For example, for the Adult Income dataset and for f1=education, f2=age, k=3 the constraint that derive from this query for instance x is
   ((y_age = age1) or (y_age = age2) or (y_age = age3)) and (y_education != x.education), where y_f is the (unknown) value of feature f of the counterfactual
   x.f is the value of feature f of instance x and age1, age2, age3 are the 3 smallest ages
*/