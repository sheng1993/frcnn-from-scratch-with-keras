import pickle

def get_data(input_path):
    with open(input_path, 'rb') as file:
        all_data = pickle.load(file)

    class_mapping = {
        'pants': 0,
        'shirt': 1,
        'jacket': 2,
        't-shirt': 3,
        'dress': 4,
        'skirt': 5,
        'cardigan': 6,
        'shorts': 7,
        'hat': 8,
        'stockings': 9,
        'coat': 10,
        'sweater': 11,
        'ruffle': 12,
        'hood': 13,
        'jumpsuit': 14
    }

    classes_count = {
        'shirt': 6161,
        't-shirt': 16550,
        'sweater': 1494,
        'cardigan': 1107,
        'jacket': 7833,
        'pants': 12415,
        'shorts': 2756,
        'skirt': 5046,
        'coat': 3124,
        'dress': 18739,
        'jumpsuit': 922,
        'hat': 2518,
        'stockings': 4326,
        'hood': 1226,
        'ruffle': 2407
    }

    return all_data, classes_count, class_mapping