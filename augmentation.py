import numpy as np
import SimpleITK as sitk
from scipy.stats import multivariate_t
from platipy.imaging.registration.utils import apply_transform

rng = np.random.RandomState()


def choose_random_direction():
    direction = rng.normal(size=3)
    direction = direction / np.linalg.norm(direction)
    return direction


def choose_random_angle(maximum_rotation=0.1):
    angle = (rng.random() * 2 - 1) * maximum_rotation * 2 * np.pi
    return angle


def choose_random_magnitude(maximum_magnitude=1):
    magnitude = rng.random() * maximum_magnitude
    return magnitude


def choose_random_vector_spherical(maximum_magnitude=1):
    direction = choose_random_direction()
    magnitude = choose_random_magnitude(maximum_magnitude)
    # print(magnitude)
    vector = direction * magnitude
    return vector


def choose_random_displacement_rectangular(maximum_displacements_voxels=(1, 1, 1)):
    displacement = rng.random(3)*np.array(maximum_displacements_voxels)
    return displacement


def convert_mm_to_voxels_3d(vector_mm, reference_image):
    spacing = reference_image.GetSpacing()
    vector_voxels = np.array(vector_mm) / spacing
    return vector_voxels


def choose_random_scale_factor(scale_factor_bounds=1.1):
    if type(scale_factor_bounds) is list or type(scale_factor_bounds) is tuple:
        scale_factor_minimum = min(scale_factor_bounds)
        scale_factor_maximum = max(scale_factor_bounds)
    else:
        if scale_factor_bounds >= 1:
            scale_factor_minimum = 1 / scale_factor_bounds
            scale_factor_maximum = scale_factor_bounds
        else:
            scale_factor_minimum = scale_factor_bounds
            scale_factor_maximum = 1 / scale_factor_bounds
    scale_factor_range = scale_factor_maximum - scale_factor_minimum
    scale_factor = rng.random() * scale_factor_range + scale_factor_minimum
    return scale_factor


def choose_random_voxel(image_shape):
    voxel = (rng.randint(image_shape[0]), rng.randint(image_shape[1]), rng.randint(image_shape[2]))
    return voxel


def generate_smooth_peaked_vector_field(image_shape, peak_voxel, peak_vector, peak_width_voxels=(1,1,1)):
    vector_field = np.empty((*image_shape, 3))
    x, y, z = np.mgrid[0:image_shape[0], 0:image_shape[1], 0:image_shape[2]]
    for axis in range(3):
        cauchy_generator = multivariate_t(loc=peak_voxel, shape=peak_width_voxels, df=1)
        vector_field[:, :, :, axis] = cauchy_generator.pdf(np.stack((x, y, z), axis=-1))
        vector_field[:, :, :, axis] *= peak_vector[axis] / cauchy_generator.pdf(peak_voxel)
    return vector_field


def generate_random_peaked_vector_field(image_shape, maximum_displacements_voxels=(10, 10, 10), peak_count=1, peak_width_voxels=(1,1,1)):
    vector_field = np.zeros((*image_shape, 3))
    for i in range(peak_count):
        peak_voxel = choose_random_voxel(image_shape)
        peak_vector = choose_random_displacement_rectangular(maximum_displacements_voxels)
        vector_field += generate_smooth_peaked_vector_field(image_shape, peak_voxel, peak_vector, peak_width_voxels)
    # print(np.max())
    return vector_field


def generate_random_vector_field_transform(reference_image, maximum_displacements_mm=(10, 10, 10), peak_count=1, peak_width_mm=(1,1,1)):
    maximum_displacements_voxels = convert_mm_to_voxels_3d(maximum_displacements_mm, reference_image)
    peak_width_voxels = convert_mm_to_voxels_3d(peak_width_mm, reference_image)
    vector_field = generate_random_peaked_vector_field(reference_image.GetSize()[::-1], maximum_displacements_voxels,
                                                       peak_count, peak_width_voxels)
    vector_image = sitk.GetImageFromArray(vector_field, isVector=True)
    # print(vector_image.GetDimension())
    # print(vector_image.GetNumberOfComponentsPerPixel())
    print(np.max(sitk.GetArrayFromImage(vector_image).flatten()))
    displacement_field = sitk.InverseDisplacementField(vector_image)
    # print(displacement_field.GetDimension())
    # print(displacement_field.GetNumberOfComponentsPerPixel())
    print(np.max(sitk.GetArrayFromImage(displacement_field).flatten()))
    displacement_field.CopyInformation(reference_image)
    # print(displacement_field.GetDimension())
    # print(displacement_field.GetNumberOfComponentsPerPixel())
    print(np.max(sitk.GetArrayFromImage(displacement_field).flatten()))
    transform = sitk.DisplacementFieldTransform(displacement_field)
    # print(transform.GetDimension())
    # transform.SetDisplacementField(displacement_field)
    return transform


def random_rotation_transform(maximum_rotation=0.1):
    axis = choose_random_direction()
    angle = choose_random_angle(maximum_rotation)
    # print(axis)
    # print(angle)
    rotation_transform = sitk.VersorTransform(axis, angle)
    return rotation_transform


def random_translation_transform(maximum_translation_mm=5):
    vector = choose_random_vector_spherical(maximum_translation_mm)
    translation_transform = sitk.TranslationTransform(3, tuple(vector))
    return translation_transform


def random_scale_transform(scale_factor_bounds=1.1):
    scale_factor = choose_random_scale_factor(scale_factor_bounds)
    scale_transform = sitk.Similarity3DTransform()
    scale_transform.SetScale(scale_factor)
    return scale_transform


def random_augmentation_transform(scale_factor_bounds=1, maximum_rotation=0, maximum_translation_mm=0):
    rotation_transform = random_rotation_transform(maximum_rotation)
    translation_transform = random_translation_transform(maximum_translation_mm)
    scale_transform = random_scale_transform(scale_factor_bounds)
    # print(rotation_transform)
    # print(translation_transform)
    composite_transform = sitk.CompositeTransform([
        scale_transform,
        rotation_transform,
        translation_transform,
    ])
    return composite_transform
    # return rotation_transform


def randomly_augment_image(image, scale_factor_bounds=1, maximum_rotation=0, maximum_translation_mm=0, default_value=0):
    transform = random_augmentation_transform(scale_factor_bounds, maximum_rotation, maximum_translation_mm)
    augmented_image = apply_transform(
        input_image=image,
        reference_image=image,
        transform=transform,
        default_value=default_value,
        interpolator=sitk.sitkLinear
    )
    return augmented_image, transform


def randomly_deform_image(image, maximum_displacements_mm=(10, 10, 10), peak_count=1, peak_width_mm=(1,1,1), default_value=0):
    if type(maximum_displacements_mm) is int or type(maximum_displacements_mm) is float:
        maximum_displacements_mm = (maximum_displacements_mm, maximum_displacements_mm, maximum_displacements_mm)
    if type(peak_width_mm) is int or type(peak_width_mm) is float:
        peak_width_mm = (peak_width_mm, peak_width_mm, peak_width_mm)
    transform = generate_random_vector_field_transform(image, maximum_displacements_mm, peak_count, peak_width_mm)
    deformed_image = apply_transform(
        input_image=image,
        reference_image=image,
        transform=transform,
        default_value=default_value,
        interpolator=sitk.sitkLinear
    )
    return deformed_image, transform
