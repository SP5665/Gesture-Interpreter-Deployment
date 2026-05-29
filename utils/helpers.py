# Function to convert gesture labels into numbers
# Machine learning models cannot understand text labels

def encode_label(label):

    # Dictionary mapping gesture → number
    label_map = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4
    }

    # Return the number for that label
    return label_map.get(label, -1)