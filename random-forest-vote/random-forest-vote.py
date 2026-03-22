import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    predictions_by_sample = np.array(predictions).T

    pred = []

    for _pred in predictions_by_sample:
        cls, counts = np.unique(_pred, return_counts=True)

        if len(set(counts)) == len(cls):
            pred.append(cls[np.argmax(counts)])
        else:
            paired = list(zip(counts, cls))
            sorted_pairs = sorted(paired)[::-1]

            if len(paired) == 1:
                pred.append(cls[0])
            else:
                largest_count = sorted_pairs[0][0]

                if sorted_pairs[1][0] != largest_count:
                    pred.append(sorted_pairs[0][1])
                else:
                    candidates = []

                    for _count,_cls in sorted_pairs:
                        if _count == largest_count:
                            candidates.append(_cls)
                        else:
                            break
                    pred.append(sorted(candidates)[0])
                    
    return pred
                        