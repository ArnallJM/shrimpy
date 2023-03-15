from pathlib import Path
import shutil


def main():
    out_dir = Path("./nnUNet_raw_data_base/nnUNet_raw_data/Task600_Sample")
    (out_dir / "imagesTr").mkdir(exist_ok = True, parents=True)
    (out_dir / "labelsTr").mkdir(exist_ok = True, parents=True)
    in_dir = Path("./DATA")

    filetype = ".nii.gz"
    image_name = "IMG_T2"
    mask_name = "MASK_TUMOUR"
    image_number = 0

    for patient in in_dir.iterdir():
        includes_all_modalities = True
        if not (patient / (image_name + filetype)).exists():
            includes_all_modalities = False
        if not (patient / (mask_name + filetype)).exists():
            includes_all_modalities = False

        if includes_all_modalities:
            source = patient / (image_name + filetype)
            destination = out_dir / "imagesTr" / f"sample_{str(image_number).zfill(4)}_0000.nii.gz"
            shutil.copy(source, destination)

            source = patient / (mask_name + filetype)
            destination = out_dir / "labelsTr" / f"sample_{str(image_number).zfill(4)}.nii.gz"
            shutil.copy(source, destination)

            image_number += 1


if __name__ == "__main__":
    main()
