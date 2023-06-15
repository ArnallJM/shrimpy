import sys
from pathlib import Path
import pathlib
import json
import os


IMAGE_MODALITIES = (
                    "IMG_T1.nii.gz",
                    "IMG_T2.nii.gz",
                    )
CONTOUR_MODALITY = "MASK_TUMOUR.nii.gz"


def patient_exists(patient_name, source_directory, number_of_augmentations):
    assert number_of_augmentations >= 0
    source_directory = Path(source_directory)
    if number_of_augmentations == 0:
        return (source_directory/patient_name).exists()

    for augmentation_number in range(number_of_augmentations):
        augmented_patient_name = patient_name + f"_AUG-{str(augmentation_number).zfill(4)}"
        if not (source_directory/augmented_patient_name).exists():
            return False
    return True


def read_patient_name_list_from_file(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file.readlines() if line.strip()]


def collate_patient_paths(patient_name_list, source_directory, number_of_augmentations):
    patient_paths_list = []
    for patient_name in patient_name_list:
        assert patient_exists(patient_name, source_directory, number_of_augmentations)
        if number_of_augmentations == 0:
            patient_paths_list.append(source_directory / patient_name)
        else:
            for augmentation_number in range(number_of_augmentations):
                augmented_patient_name = patient_name + f"_AUG-{str(augmentation_number).zfill(4)}"
                patient_paths_list.append(source_directory / augmented_patient_name)
    return patient_paths_list


def create_raw_database(patient_paths_list, task_directory, task_name=None, test=False):
    if task_name is None:
        task_name = "GENERIC"
    task_directory = Path(task_directory)
    images_directory = task_directory/"imagesTs" if test else task_directory/"imagesTr"
    images_directory.mkdir(parents=True, exist_ok=True)
    labels_directory = None if test else task_directory/"labelsTr"
    if not test:
        labels_directory.mkdir(parents=True, exist_ok=True)
    datum_dicts = []
    for patient_number in range(len(patient_paths_list)):
        for modality_number in range(len(IMAGE_MODALITIES)):
            source = patient_paths_list[patient_number]/IMAGE_MODALITIES[modality_number]
            destination = images_directory/f"{task_name}_{str(patient_number).zfill(4)}_{str(modality_number).zfill(4)}.nii.gz"
            destination.absolute().symlink_to(source.absolute())
            # os.symlink(source.absolute(), destination.absolute())
        if test:
            datum_dict = f"./imagesTr/{task_name}_{str(patient_number).zfill(4)}.nii.gz"
        else:
            source = patient_paths_list[patient_number]/CONTOUR_MODALITY
            destination = labels_directory/f"{task_name}_{str(patient_number).zfill(4)}.nii.gz"
            destination.absolute().symlink_to(source.absolute())
            # os.symlink(source, destination)
            datum_dict = {"image": f"./imagesTr/{task_name}_{str(patient_number).zfill(4)}.nii.gz",
                          "label": f"./labelsTr/{task_name}_{str(patient_number).zfill(4)}.nii.gz"}
        datum_dicts.append(datum_dict)
    return datum_dicts


def create_database_json(task_directory, train_datum_dicts, test_datum_dicts=None, task_name=None, task_description=None):
    if task_name is None:
        task_name = "GENERIC"
    if task_description is None:
        task_description = "GENERIC DESCRIPTION: training nnUNet on augmented data"
    if test_datum_dicts is None:
        test_datum_dicts = []
    data_dict = {
        "name": task_name,
        "description": task_description,
        "reference": "USYD",
        "licence": "CC-BY-SA 4.0",
        "relase": "0.1",
        "tensorImageSize": "3D",
        "modality": {
            "0": "IMG_T1",
            "1": "IMG_T2",
        },
        "labels": {
            "0": "background",
            "1": "OBJECT",
        },
        "numTraining": len(train_datum_dicts),
        "numTest": len(test_datum_dicts),
        "training": train_datum_dicts,
        "test": test_datum_dicts
    }
    with open(task_directory / 'dataset.json', 'w') as file:
        json.dump(data_dict, file)


def generate_database(train_patient_name_list, test_patient_name_list, train_directory, test_directory, task_directory, number_of_augmentations, task_name=None, task_description=None):
    number_of_augmentations = int(number_of_augmentations)
    assert number_of_augmentations >= 0
    train_directory = Path(train_directory)
    test_directory = Path(test_directory)
    task_directory = Path(task_directory)
    if isinstance(train_patient_name_list, (str, pathlib.PurePath)):
        train_patient_name_list = read_patient_name_list_from_file(train_patient_name_list)
    if isinstance(test_patient_name_list, (str, pathlib.PurePath)):
        test_patient_name_list = read_patient_name_list_from_file(test_patient_name_list)
    train_patient_paths_list = collate_patient_paths(train_patient_name_list, train_directory, number_of_augmentations)
    test_patient_paths_list = collate_patient_paths(test_patient_name_list, test_directory, 0)
    train_datum_dicts = create_raw_database(train_patient_paths_list, task_directory, task_name=task_name, test=False)
    test_datum_dicts = create_raw_database(test_patient_paths_list, task_directory, task_name=task_name, test=True)
    create_database_json(task_directory, train_datum_dicts, test_datum_dicts, task_name, task_description)


def read_and_generate_experiment(experiment_file):
    with open(experiment_file, 'r') as file:
        args = [line.strip() for line in file.readlines()]
    assert len(args) >= 6
    # print(args)
    task_name = None
    task_description = None
    if len(args) >= 7:
        task_name = args[6]
    if len(args) >= 8:
        task_description = args[7]
    # arg_length = min(len(args), 8)
    # args = args[:arg_length]
    # print(args[:8])
    generate_database(*args[:6], task_name=task_name, task_description=task_description)


def main(experiment_file):
    read_and_generate_experiment(experiment_file)


if __name__ == "__main__":
    assert len(sys.argv) >= 2
    main(sys.argv[1])



