import numpy as np
import SimpleITK as sitk
from pathlib import Path
from platipy.imaging.registration.utils import apply_transform
from .augmentation import random_augmentation_transform, generate_random_vector_field_transform, full_kernel, convert_mm_to_voxels_3d
from .intensity import randomly_augment_image_intensity
from .rng import rng
import warnings

SEGMENTATION_FILENAME = "MASK_TUMOUR.nii.gz"
REFERENCE_IMAGE_MODALITY = "IMG_T1.nii.gz"
DEFAULT_VOXEL_VALUE = 0
# DISPLACEMENT_FIELD_PEAK_WIDTH_MM = 50
AUGMENTATION_TRANSFORMS = (
    lambda *args: generate_random_vector_field_transform(args[0],
                                                         maximum_displacements_mm=2, peak_count=1000,
                                                         peak_width_mm=50),
    lambda *args: random_augmentation_transform(scale_factor_bounds=1.1, maximum_rotation=0.01,
                                                maximum_translation_mm=10),
)
INTENSITY_AUGMENTATION = {
    "IMG_T1.nii.gz": lambda image: randomly_augment_image_intensity(image, scale_factor_bounds=1.2, noise_sigma=10),
    "IMG_T2.nii.gz": lambda image: randomly_augment_image_intensity(image, scale_factor_bounds=1.2, noise_sigma=10),
    "MASK_TUMOUR.nii.gz": lambda image: image,
}


# INTENSITY_AUGMENTATION = lambda image: randomly_augment_image_intensity(image, scale_factor_bounds=2, noise_sigma=0.2)
# def INTENSITY_AUGMENTATION(image): return randomly_augment_image_intensity(image, scale_factor_bounds=2, noise_sigma=0.2)


def read_patient(patient_directory):
    patient_directory = Path(patient_directory)
    assert ((patient_directory / SEGMENTATION_FILENAME).exists())
    patient_name = patient_directory.name
    patient_dictionary = {}
    for image_modality in patient_directory.iterdir():
        image = sitk.ReadImage(str(image_modality), sitk.sitkFloat32)
        patient_dictionary[image_modality.name] = image
    return patient_name, patient_dictionary


def apply_transform_to_patient(transform, patient_dictionary):
    transformed_patient_dictionary = {}
    for modality in patient_dictionary.keys():
        image = patient_dictionary[modality]
        if modality == SEGMENTATION_FILENAME:
            interpolator = sitk.sitkNearestNeighbor
        else:
            interpolator = sitk.sitkLinear
        transformed_image = apply_transform(
            input_image=image,
            reference_image=image,
            transform=transform,
            default_value=DEFAULT_VOXEL_VALUE,
            interpolator=interpolator
        )
        transformed_patient_dictionary[modality] = transformed_image
    return transformed_patient_dictionary


def augment_patient_intensity(patient_dictionary):
    augmented_patient_dictionary = {}
    for modality in patient_dictionary.keys():
        image = patient_dictionary[modality]
        augmented_image = INTENSITY_AUGMENTATION[modality](image)
        augmented_patient_dictionary[modality] = augmented_image
    return augmented_patient_dictionary


def augment_patient(patient_dictionary):
    # TODO decide whether to transform or change intensity first
    reference_image = patient_dictionary[REFERENCE_IMAGE_MODALITY]
    augmentation_transforms = [augmentation_function(reference_image) for augmentation_function in AUGMENTATION_TRANSFORMS]
    composite_transform = sitk.CompositeTransform(augmentation_transforms)
    transformed_patient_dictionary = apply_transform_to_patient(composite_transform, patient_dictionary)
    augmented_patient_dictionary = augment_patient_intensity(transformed_patient_dictionary)
    return augmented_patient_dictionary


def write_patient(patient_dictionary, patient_directory):
    location = Path(patient_directory)
    location.mkdir(parents=True, exist_ok=True)
    for modality in patient_dictionary:
        image = patient_dictionary[modality]
        sitk.WriteImage(image, str(location / modality))


def read_augment_write(patient_directory, augmentations_directory, number_of_augmentations=1, replace_existing=True):
    patient_name, patient_dictionary = read_patient(patient_directory)
    assert (number_of_augmentations <= 1000)
    for augmentation_number in range(number_of_augmentations):
        augmented_patient_name = patient_name + f"_AUG-{str(augmentation_number).zfill(4)}"
        augmented_patient_directory = Path(augmentations_directory) / augmented_patient_name
        if not replace_existing and augmented_patient_directory.exists():
            continue
        augmented_patient_dictionary = augment_patient(patient_dictionary)
        write_patient(augmented_patient_dictionary, augmented_patient_directory)


# def find_largest_reference_image(patients_directory):
#     patients_directory = Path(patients_directory)
#     image_volume_voxels = 0
#     image = None
#     for patient_directory in patients_directory.iterdir():
#         if patient_directory.is_dir():
#             if (patient_directory / SEGMENTATION_FILENAME).exists():
#                 reference_image = sitk.ReadImage(str(patient_directory / REFERENCE_IMAGE_MODALITY))
#                 reference_shape = np.array(reference_image.GetSize())
#                 reference_volume = np.product(reference_shape)
#                 if reference_volume > image_volume_voxels:
#                     image_volume_voxels = reference_volume
#                     image = reference_image
#     return image
#
#
# def precalculate_kernel(patients_directory):
#     reference_image = find_largest_reference_image(patients_directory)
#     peak_width_voxels = convert_mm_to_voxels_3d(DISPLACEMENT_FIELD_PEAK_WIDTH_MM, reference_image)
#     full_kernel(reference_image.GetSize(), peak_width_voxels)


def read_augment_write_directory(patients_directory, augmentations_directory, number_of_augmentations_per_patient=None,
                                 total_number_of_augmentations=None, replace_existing=True):
    if total_number_of_augmentations is not None and number_of_augmentations_per_patient is not None:
        warnings.warn("total_number_of_augmentations overrides number_of_augmentations_per_patient")
    elif number_of_augmentations_per_patient is None:
        number_of_augmentations_per_patient = 1
    patients_directory = Path(patients_directory)
    augmentations_directory = Path(augmentations_directory)
    patient_directories = []
    for patient_directory in patients_directory.iterdir():
        if patient_directory.is_dir():
            if (patient_directory / SEGMENTATION_FILENAME).exists():
                patient_directories.append(patient_directory)
    if total_number_of_augmentations is not None:
        number_of_augmentations_per_patient = rng.multinomial(len(patient_directories),
                                                              [1 / len(patient_directories)] * len(patient_directories))
    else:
        number_of_augmentations_per_patient = [number_of_augmentations_per_patient] * len(patient_directories)

    for i in range(len(patient_directories)):
        read_augment_write(patient_directories[i], augmentations_directory, number_of_augmentations_per_patient[i], replace_existing)
