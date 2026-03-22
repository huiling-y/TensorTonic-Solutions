import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    
    pred = []

    for _pred in zip(*predictions):
        cls, counts = np.unique(_pred, return_counts=True)

        max_count = max(counts)

        pred.append(cls[counts == max_count][0])


                    
    return pred
                        