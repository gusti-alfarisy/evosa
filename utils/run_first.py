import os

def init_directories():
    dataset_dir = os.path.join("dataset")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    output1_dir = os.path.join("output", "logs")
    if not os.path.exists(output1_dir):
        os.makedirs(output1_dir)

    output2_dir = os.path.join("output", "saved_objects")
    if not os.path.exists(output2_dir):
        os.makedirs(output2_dir)

    output_json = os.path.join("output", "json_bo")
    if not os.path.exists(output_json):
        os.makedirs(output_json)
