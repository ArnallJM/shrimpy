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


def choose_random_vector(maximum_magnitude=1):
    direction = choose_random_direction()
    magnitude = choose_random_magnitude(maximum_magnitude)
    # print(magnitude)
    vector = direction * magnitude
    return vector


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


def generate_smooth_peaked_vector_field(image_shape, peak_voxel, peak_vector, scale=1):
    vector_field = np.empty((*image_shape, 3))
    x, y, z = np.mgrid[0:image_shape[0], 0:image_shape[1], 0:image_shape[2]]
    for axis in range(3):
        cauchy_generator = multivariate_t(loc=peak_voxel, shape=scale, df=1)
        vector_field[:, :, :, axis] = cauchy_generator.pdf(np.stack((x, y, z), axis=-1))
        vector_field[:, :, :, axis] *= peak_vector[axis] / cauchy_generator.pdf(peak_voxel)
    return vector_field


def generate_random_peaked_vector_field(image_shape, maximum_magnitude_voxels=10, peak_count=1, scale=1):
    vector_field = np.zeros((*image_shape, 3))
    for i in range(peak_count):
        peak_voxel = choose_random_voxel(image_shape)
        peak_vector = choose_random_vector(maximum_magnitude_voxels)
        vector_field += generate_smooth_peaked_vector_field(image_shape, peak_voxel, peak_vector, scale)
    return vector_field


def generate_random_vector_field_transform(reference_image, maximum_magnitude_voxels=10, peak_count=1, scale=1):
    vector_field = generate_random_peaked_vector_field(reference_image.GetSize()[::-1], maximum_magnitude_voxels,
                                                       peak_count, scale)
    vector_image = sitk.GetImageFromArray(vector_field, isVector=True)
    displacement_field = sitk.InvertDisplacementField(vector_image)
    displacement_field.CopyInformation(reference_image)
    transform = sitk.DisplacementFieldTransform(displacement_field)
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
    vector = choose_random_vector(maximum_translation_mm)
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


def randomly_deform_image(image, maximum_magnitude_voxels=10, peak_count=1, scale=1, default_value=0):
    transform = generate_random_vector_field_transform(image, maximum_magnitude_voxels, peak_count, scale)
    deformed_image = apply_transform(
        input_image=image,
        reference_image=image,
        transform=transform,
        default_value=default_value,
        interpolator=sitk.sitkLinear
    )
    return deformed_image, transform
