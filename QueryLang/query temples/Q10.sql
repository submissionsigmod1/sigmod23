SELECT C.*
FROM CFs as C
WHERE C.F1 = (SELECT F1
			  FROM (SELECT F1,
                           ROW_NUMBER() OVER (ORDER BY COUNT(*) ASC) Freq
			        FROM Instances
					GROUP BY F1)
			  WHERE Freq = 1)



/* A query looking for CFs that change feature f1 to the least frequent value of feature f1 in the dataset.
    For example, for the Adult Income dataset and for f1=education the constraint that derive from this query for instance x is
   (y_ education =  education1), where y_f is the (unknown) value of feature f of the counterfactual
    education1 is the least frequent education status
*/

