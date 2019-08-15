import pickle

def get_data(input_path):
    with open(input_path, 'rb') as file:
        all_data = pickle.load(file)

    class_mapping = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        6: 5,
        7: 6,
        8: 7,
        9: 8,
        10: 9,
        11: 10,
        14: 11,
        21: 12,
        27: 13,
        43: 14
    }

    classes_count = {
        0: 6161,
        1: 16550,
        2: 1494,
        3: 1107,
        4: 7833,
        6: 12415,
        7: 2756,
        8: 5046,
        9: 3124,
        10: 18739,
        11: 922,
        14: 2518,
        21: 4326,
        27: 1226,
        43: 2407
    }

    return all_data, classes_count, class_mapping