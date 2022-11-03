SELECT *
FROM CFs AS C
WHERE C.F1 BETWEEN (SELECT AVG(F1)
                   FROM Instances)
                   AND
                   (SELECT 0.5*MAX(F1)
                   FROM Instances)



/* A query looking for CFs that feature f1 o be between the average and the half of the maximum value of feature f1 in the dataset.
    For example, for the Adult Income dataset and for f1=education the constraint that derive from this query for instance x is
   (y_ age >  age1) and (y_age <  age2), where y_f is the (unknown) value of feature f of the counterfactual
    age1 is the average age and age2 is half of the max age
*/
