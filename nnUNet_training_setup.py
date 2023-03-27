from pathlib import Path
import shutil
import warnings
import json


MASK_NAME = "MASK_TUMOUR"
FILETYPE = ".nii.gz"


def modalities_exist(patient, image_names):
    for image_name in image_names:
        if not (patient / (image_name + FILETYPE)).exists():
            return False
    return True


def create_experiment_raw_data_base(experiment_name="600_unaugmented5"):
    assert experiment_name[3] == "_" and 600 <= int(experiment_name[:3]) < 700
    out_dir = Path(f"./nnUNet_raw_data_base/nnUNet_raw_data/Task{experiment_name}")
    (out_dir / "imagesTr").mkdir(exist_ok=True, parents=True)
    (out_dir / "labelsTr").mkdir(exist_ok=True, parents=True)
    in_dir = Path("./DATA")

    with open(f"./experiments/{experiment_name}.txt") as file:
        patient_ids = file.readlines()

    image_names = ("IMG_T1", "IMG_T2")
    image_number = 0
    failed_patients = []

    # for patient in in_dir.iterdir():
    #     if not modalities_exist(patient, image_names):
    #         failed_patients.append(patient.name)
    #         continue
    #
    #     for modality_number in range(len(image_names)):
    #         source = patient / (image_names[modality_number] + FILETYPE)
    #         destination = out_dir / "imagesTr" / f"sample_{str(image_number).zfill(4)}_{str(modality_number).zfill(4)}.nii.gz"
    #         shutil.copy(source, destination)
    #
    #     source = patient / (MASK_NAME + FILETYPE)
    #     destination = out_dir / "labelsTr" / f"sample_{str(image_number).zfill(4)}.nii.gz"
    #     shutil.copy(source, destination)
    #     image_number += 1
    for patient_id in patient_ids:
        patient = in_dir / f"ACRIN-6698-{patient_id.strip()}"
        if not modalities_exist(patient, image_names):
            failed_patients.append(patient.name)
            continue
        for modality_number in range(len(image_names)):
            source = patient / (image_names[modality_number] + FILETYPE)
            destination = out_dir / "imagesTr" / f"sample_{str(image_number).zfill(4)}_{str(modality_number).zfill(4)}.nii.gz"
            shutil.copy(source, destination)
        source = patient / (MASK_NAME + FILETYPE)
        destination = out_dir / "labelsTr" / f"sample_{str(image_number).zfill(4)}.nii.gz"
        shutil.copy(source, destination)
        image_number += 1

    if len(failed_patients) > 0:
        warnings.warn(f"The following patient(s) did not have all the required modalities: {str(failed_patients)}")

    data_dict = {
        "name": "UNAUGMENTED5",
        "description": "TESTING ISPY NON-AUGMENTED SEGMENTATION",
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
        "numTraining": image_number,
        "numTest": 0,
        "training": [
            {
                "image": f"./imagesTr/sample_{str(i).zfill(4)}.nii.gz",
                "label": f"./labelsTr/sample_{str(i).zfill(4)}.nii.gz"
            }
            for i in range(image_number)
        ],
        "test": []
    }

    with open(out_dir / 'dataset.json', 'w') as fp:
        json.dump(data_dict, fp)


if __name__ == "__main__":
    create_experiment_raw_data_base()
