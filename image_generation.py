import numpy as np
import SimpleITK as sitk
from pathlib import Path
from platipy.imaging.registration.utils import apply_transform
from platipy.imaging import ImageVisualiser
from platipy.imaging.label.utils import get_com
from .augmentation import random_augmentation_transform,  generate_random_vector_field_transform
from .augmentation import random_rotation_transform, random_translation_transform, random_scale_transform
from .pipeline import read_augment_write, read_augment_write_directory
from .intensity import randomly_augment_image_intensity, apply_noise
from .pipeline import read_patient
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import warnings


def visualise_image(image, transform=None, contour=None, save_name=None, vector_field=None):
    if type(image) == str or type(image) == Path:
        image = sitk.ReadImage(str(image))
    if type(contour) == str or type(contour) == Path:
        contour = sitk.ReadImage(str(contour))
    if type(vector_field) == str or type(vector_field) == Path:
        vector_field = sitk.ReadImage(str(vector_field))
    if transform is not None:
        image = apply_transform(image, image, transform, interpolator=sitk.sitkLinear)
    if contour is not None:
        if transform is not None:
            contour = apply_transform(contour, contour, transform, interpolator=sitk.sitkNearestNeighbor)
        contour_com = get_com(contour)[0]
    else:
        contour_com = None
    vis = ImageVisualiser(image, cut=contour_com, axis='z', figure_size_in=5)
    if contour is not None:
        vis.add_contour(contour)
    if vector_field is not None:
        vis.add_vector_overlay(vector_field, arrow_scale=1, subsample=32, show_colorbar=False, color_function='perpendicular', colormap="Wistia", arrow_width=2)
    fig = vis.show()
    if save_name is not None:
        fig.savefig(save_name)
    return fig


def compare_image_histograms(image1, image2, legend=None, save_name=None):
    hist1, bins1 = image_histogram(image1)
    hist2, bins2 = image_histogram(image2)
    fig, ax = plt.subplots()
    line1, = ax.loglog(bins1, hist1)
    line2, = ax.loglog(bins2, hist2)
    if legend is not None:
        assert(len(legend) == 2)
        ax.legend([line1, line2], legend)
    plt.xlabel("Voxel intensity")
    plt.ylabel("Frequency")
    if save_name is not None:
        fig.savefig(save_name)
    # fig.show()
    return fig


def image_histogram(image, n_bins=100):
    array = sitk.GetArrayFromImage(image).flatten()
    hist, bins = np.histogram(array, bins=n_bins, range=(0, 2000), density=True)
    bin_centers = (bins[1:]+bins[:-1])/2
    return hist, bin_centers


def patient_name_augmentation(filename):
    filename = str(filename)
    split = filename.split('_')
    if len(split) == 1:
        return split[0], None
    else:
        return split[:2]


def list_distinct_patient_names(directory):
    patients_list = []
    directory = Path(directory)
    for patient in directory.iterdir():
        patient_name, _ = patient_name_augmentation(patient.name)
        if patient_name not in patients_list:
            patients_list.append(patient_name)
    return patients_list


def create_augmentation_visualisaitons(augmentations_directory, visualisations_directory, original_directory, augmentations_count=5, patients_count=5):
    augmentations_directory = Path(augmentations_directory)
    visualisations_directory = Path(visualisations_directory)
    original_directory = Path(original_directory)
    visualisations_directory.mkdir(parents=True, exist_ok=True)
    image_T1_name = "IMG_T1.nii.gz"
    image_T2_name = "IMG_T2.nii.gz"
    contour_name = "MASK_TUMOUR.nii.gz"
    patient_names = list_distinct_patient_names(augmentations_directory)
    if len(patient_names) < patients_count:
        warnings.warn(f"Only {len(patient_names)} patient(s) found.")
        patients_count = len(patient_names)
    for i in range(patients_count):
        patient_name = patient_names[i]
        for augmentation_number in range(augmentations_count):
            augmentation_name = f"{patient_name}_AUG-{str(augmentation_number).zfill(4)}"
            T1_filename = str((augmentations_directory / augmentation_name) / image_T1_name)
            T2_filename = str((augmentations_directory / augmentation_name) / image_T2_name)
            contour_filename = str(augmentations_directory / augmentation_name / contour_name)
            try:
                fig = visualise_image(T1_filename, contour=contour_filename, save_name=visualisations_directory/f"{augmentation_name}_T1.png")
                plt.close(fig)
                fig = visualise_image(T2_filename, contour=contour_filename, save_name=visualisations_directory/f"{augmentation_name}_T2.png")
                plt.close(fig)
            except FileNotFoundError:
                continue
            print(augmentation_name)


def create_image_visualisations(patients_directory, visualisations_directory):
    patients_directory = Path(patients_directory)
    visualisations_directory = Path(visualisations_directory)
    visualisations_directory.mkdir(parents=True, exist_ok=True)
    image_T1_name = "IMG_T1.nii.gz"
    image_T2_name = "IMG_T2.nii.gz"
    contour_name = "MASK_TUMOUR.nii.gz"
    for patient_directory in patients_directory.iterdir():
        patient_name, patient_dictionary = read_patient(patient_directory)
        fig = visualise_image(patient_dictionary[image_T1_name], contour=patient_dictionary[contour_name], save_name=visualisations_directory/f"{patient_name}_T1.png")
        plt.close(fig)
        fig = visualise_image(patient_dictionary[image_T2_name], contour=patient_dictionary[contour_name], save_name=visualisations_directory/f"{patient_name}_T2.png")
        plt.close(fig)
        print(patient_name)

def collate_augmentation_visualisations(visualisations_directory, images=5, one_patient=None, original_directory=None, save_name=None):
    if (one_patient is None) and (original_directory is not None):
        warnings.warn("Cannot include original for multi-patient visualisation.")
    visualisations_directory = Path(visualisations_directory)
    patients_names = list_distinct_patient_names(visualisations_directory)
    figure_height_in = 6
    fig, axs = plt.subplots(2, 5, sharex='all', sharey='all', figsize=(figure_height_in/2*images, figure_height_in))
    image_number = 0
    if original_directory is not None and one_patient is not None:
        original_directory = Path(original_directory)
        image_T1 = mpimg.imread(original_directory / f"{one_patient}_T1.png")
        image_T2 = mpimg.imread(original_directory / f"{one_patient}_T2.png")
        axs[0,0].imshow(image_T1)
        # axs[0,0].set_axis_off()
        axs[0,0].tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        axs[1,0].imshow(image_T2)
        # axs[1,0].set_axis_off()
        axs[1,0].tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        axs[1,0].set_xlabel("Original", fontsize='large')
        image_number += 1
    while image_number < images:
        if one_patient is None:
            try:
                image_prefix = f"{patients_names[image_number]}_AUG-0000"
            except IndexError:
                break
        else:
            image_prefix = f"{one_patient}_AUG-{str(image_number-int(original_directory is not None)).zfill(4)}"
        try:
            image_T1 = mpimg.imread(visualisations_directory / f"{image_prefix}_T1.png")
            image_T2 = mpimg.imread(visualisations_directory / f"{image_prefix}_T2.png")
            axs[0,image_number].imshow(image_T1)
            # axs[0,image_number].set_axis_off()
            axs[0,image_number].tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            axs[1,image_number].imshow(image_T2)
            # axs[1,image_number].set_axis_off()
            axs[1,image_number].tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            axs[1, image_number].set_xlabel(f"Augmentation {image_number-int(original_directory is not None)+1}", fontsize='large')
        except FileNotFoundError:
            break
        image_number += 1
    axs[0,0].set_ylabel("T1", fontsize='large')
    axs[1,0].set_ylabel("T2", fontsize='large')
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.show()
    if save_name is not None:
        fig.savefig(save_name, bbox_inches='tight')
    return fig



