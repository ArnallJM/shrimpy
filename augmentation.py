import numpy as np
import SimpleITK as sitk
from scipy.stats import multivariate_t
from scipy.ndimage import convolve
from tqdm.auto import tqdm
# from platipy.imaging.registration.utils import apply_transform
from .rng import rng
import time


KERNEL_CACHE = {
    "kernel": None,
    "image_shape": None,
    "distribution_shape": None,
    "distribution_function": None,
}


# Taken from platipy
def apply_transform(
    input_image,
    reference_image=None,
    transform=None,
    default_value=0,
    interpolator=sitk.sitkNearestNeighbor,
):
    """
    Transform a volume of structure with the given deformation field.

    Args
        input_image (SimpleITK.Image): The image, to which the transform is applied
        reference_image (SimpleITK.Image): The image will be resampled into this reference space.
        transform (SimpleITK.Transform): The transformation
        default_value: Default (background) value. Defaults to 0.
        interpolator (int, optional): The interpolation order.
                                Available options:
                                    - SimpleITK.sitkNearestNeighbor
                                    - SimpleITK.sitkLinear
                                    - SimpleITK.sitkBSpline
                                Defaults to SimpleITK.sitkNearestNeighbor

    Returns
        (SimpleITK.Image): the transformed image

    """
    original_image_type = input_image.GetPixelID()

    resampler = sitk.ResampleImageFilter()

    if reference_image:
        resampler.SetReferenceImage(reference_image)
    else:
        resampler.SetReferenceImage(input_image)

    if transform:
        resampler.SetTransform(transform)

    resampler.SetDefaultPixelValue(default_value)
    resampler.SetInterpolator(interpolator)

    output_image = resampler.Execute(input_image)
    output_image = sitk.Cast(output_image, original_image_type)

    return output_image


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
    displacement = (2*rng.random(3)-1) * np.array(maximum_displacements_voxels)
    return displacement


def convert_mm_to_voxels_3d(vector_mm, reference_image):
    spacing = reference_image.GetSpacing()[::-1]
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
    voxel = (rng.integers(image_shape[0]), rng.integers(image_shape[1]), rng.integers(image_shape[2]))
    return voxel


def gaussian(voxel, loc=None, shape=1):
    if loc is None:
        loc = np.zeros(3)
    return np.exp(-np.sum(((voxel - loc)/shape)**2, axis=-1)/2)


def cauchy(voxel, loc=None, shape=1):
    if loc is None:
        loc = np.zeros(3)
    cauchy_generator = multivariate_t(loc=loc, shape=shape, df=1)
    return cauchy_generator.pdf(voxel) / cauchy_generator.pdf(loc)


def cached_kernel_is_fine(image_shape, distribution_shape, distribution_function):
    for value in KERNEL_CACHE.values():
        if value is None:
            return False
    if np.any(image_shape > KERNEL_CACHE["image_shape"]):
        return False
    if np.any(distribution_shape != KERNEL_CACHE["distribution_shape"]):
        # print(distribution_shape, KERNEL_CACHE["distribution_shape"])
        return False
    if distribution_function != KERNEL_CACHE["distribution_function"]:
        # print("mismatching distribution functions")
        return False
    return True


def full_kernel(image_shape, distribution_shape=1, distribution_function=gaussian):
    if cached_kernel_is_fine(image_shape, distribution_shape, distribution_function):
        return KERNEL_CACHE["kernel"]
    print("generating and caching new kernel")
    KERNEL_CACHE["kernel"] = None
    image_shape = np.array(image_shape)
    full_kernel_shape = image_shape*2-1
    x, y, z = np.mgrid[0:full_kernel_shape[0], 0:full_kernel_shape[1], 0:full_kernel_shape[2]]
    voxels = np.stack((x, y, z), axis=-1)
    kernel = distribution_function(voxels, loc=image_shape-1, shape=distribution_shape)
    KERNEL_CACHE["kernel"] = kernel
    KERNEL_CACHE["image_shape"] = image_shape
    KERNEL_CACHE["distribution_shape"] = distribution_shape
    KERNEL_CACHE["distribution_function"] = distribution_function
    return kernel


def fit_kernel(image_shape, kernel, loc):
    assert(len(image_shape) == 3)
    # print(kernel.shape)
    kernel_center = (np.array(kernel.shape, dtype=int)+1)//2
    # print(kernel_center)
    lower_bound = kernel_center-loc-1
    upper_bound = kernel_center + image_shape-loc-1
    fitted_kernel = kernel[lower_bound[0]:upper_bound[0], lower_bound[1]:upper_bound[1], lower_bound[2]:upper_bound[2]]
    return fitted_kernel


def generate_smooth_peaked_vector_field(image_shape, peak_voxel, peak_vector, peak_width_voxels=1, kernel="gaussian"):
    vector_field = np.empty((*image_shape, 3))
    x, y, z = np.mgrid[0:image_shape[0], 0:image_shape[1], 0:image_shape[2]]
    for axis in range(3):
        if kernel == "cauchy":
            # cauchy_generator = multivariate_t(loc=peak_voxel, shape=peak_width_voxels, df=1)
            # vector_field[:, :, :, axis] = cauchy_generator.pdf(np.stack((x, y, z), axis=-1))
            # vector_field[:, :, :, axis] *= peak_vector[axis] / cauchy_generator.pdf(peak_voxel)

            vector_field[:,:,:, axis] = cauchy(np.stack((x, y, z), axis=-1), loc=peak_voxel, shape=peak_width_voxels)*peak_vector[axis]
        elif kernel == "gaussian":
            vector_field[:,:,:, axis] = gaussian(np.stack((x, y, z), axis=-1), loc=peak_voxel, shape=peak_width_voxels)*peak_vector[axis]
            # raise NotImplementedError()
        else:
            raise ValueError("kernel must be gaussian or cauchy")
    return vector_field


def alt_generate_random_peaked_vector_field(image_shape, maximum_displacements_voxels=(10, 10, 10), peak_count=1,
                                        peak_width_voxels=(1, 1, 1),  kernel_size_factor=3):
    now = time.time()
    vector_field = np.zeros((*image_shape, 3))
    for i in range(peak_count):
        # print(f"Calculating peak {i}")
        peak_voxel = choose_random_voxel(image_shape)
        peak_vector = choose_random_displacement_rectangular(maximum_displacements_voxels)
        # print(peak_voxel)
        vector_field[peak_voxel[0], peak_voxel[1], peak_voxel[2], :] = peak_vector
    kernel_shape = np.array(peak_width_voxels, dtype=int)*kernel_size_factor
    for i in range(3):
        if kernel_shape[i] > image_shape[i]:
            kernel_shape[i] = image_shape[i]
    print(kernel_shape)
    x, y, z = np.mgrid[0:kernel_shape[0], 0:kernel_shape[1], 0:kernel_shape[2]]
    kernel_generator = multivariate_t(loc=kernel_shape/2, shape=peak_width_voxels, df=1)
    kernel = kernel_generator.pdf(np.stack((x,y,z), axis=-1))
    smoothed_vector_field = np.empty_like(vector_field)
    for axis in range(3):
        print(f"Convolving axis {axis}")
        smoothed_vector_field[:,:,:,axis] = convolve(vector_field[:,:,:,axis], kernel)
    print(time.time()-now)
    return smoothed_vector_field


def precalculated_generate_smooth_peaked_vector_field(image_shape, kernel, peak_voxel, peak_vector):
    # vector_field = np.empty((*image_shape, 3))
    fitted_kernel = fit_kernel(image_shape, kernel, peak_voxel)
    vector_field = fitted_kernel.reshape(*image_shape, 1)*peak_vector.reshape(1,1,1,-1)
    # print(vector_field.shape)
    return vector_field


def precalculated_generate_random_peaked_vector_field(image_shape, maximum_displacements_voxels=(10,10,10), peak_count=1, peak_width_voxels=1, kernel="gaussian", verbose=False):
    # now = time.time()
    image_shape = np.array(image_shape)
    vector_field = np.zeros((*image_shape, 3))
    if type(kernel) == str:
        if kernel == "gaussian":
            distribution_function = gaussian
        elif kernel == "cauchy":
            distribution_function = cauchy
        else:
            raise ValueError("kernel must be gaussian or cauchy")
        kernel = full_kernel(image_shape, distribution_shape=peak_width_voxels, distribution_function=distribution_function)
    if verbose:
        print("generating peaks:")
        iterator = tqdm(range(peak_count))
    else:
        iterator = range(peak_count)
    for i in iterator:
        # print(f"Calculating peak {i}")
        peak_voxel = choose_random_voxel(image_shape)
        peak_vector = choose_random_displacement_rectangular(maximum_displacements_voxels)
        # print(peak_vector)
        vector_field += precalculated_generate_smooth_peaked_vector_field(image_shape, kernel, peak_voxel, peak_vector)
    # print(time.time()-now)
    return vector_field


def consistent_vector_field(image_shape, peak_vectors_voxels, peak_voxels, peak_width_voxels=1, kernel="gaussian"):
    image_shape = np.array(image_shape)
    vector_field = np.zeros((*image_shape, 3))
    if kernel == "gaussian":
        distribution_function = gaussian
    elif kernel == "cauchy":
        distribution_function = cauchy
    else:
        raise ValueError("kernel must be gaussian or cauchy")
    kernel = full_kernel(image_shape, distribution_shape=peak_width_voxels, distribution_function=distribution_function)
    print("generating peaks:")
    for i in tqdm(range(len(peak_vectors_voxels))):
        # print(f"Calculating peak {i}")
        peak_voxel = peak_voxels[i]
        peak_vector = peak_vectors_voxels[i]
        # print(peak_vector)
        # print(peak_vector)
        vector_field += precalculated_generate_smooth_peaked_vector_field(image_shape, kernel, peak_voxel, peak_vector)
    # print(time.time()-now)
    return vector_field


def generate_random_peaked_vector_field(image_shape, maximum_displacements_voxels=(10, 10, 10), peak_count=1,
                                        peak_width_voxels=1):
    now = time.time()
    vector_field = np.zeros((*image_shape, 3))
    for i in range(peak_count):
        print(f"Calculating peak {i}")
        peak_voxel = choose_random_voxel(image_shape)
        peak_vector = choose_random_displacement_rectangular(maximum_displacements_voxels)
        # print(peak_vector)
        vector_field += generate_smooth_peaked_vector_field(image_shape, peak_voxel, peak_vector, peak_width_voxels)
    # print(np.max())
    print(time.time()-now)
    return vector_field


def consistent_vector_field_transform(reference_image, peak_vectors_mm, peak_voxels, peak_width_mm=1):
    peak_vectors_voxels = [convert_mm_to_voxels_3d(peak_vector_mm, reference_image)[::-1] for peak_vector_mm in peak_vectors_mm]
    peak_width_voxels = convert_mm_to_voxels_3d(peak_width_mm, reference_image)
    displacement_field_array = consistent_vector_field(reference_image.GetSize()[::-1], peak_vectors_voxels, peak_voxels, peak_width_voxels)
    displacement_field = sitk.GetImageFromArray(displacement_field_array, isVector=True)
    displacement_field.CopyInformation(reference_image)
    transform = sitk.DisplacementFieldTransform(displacement_field)
    return transform


def generate_random_vector_field_transform(reference_image, maximum_displacements_mm=(10, 10, 10), peak_count=1,
                                           peak_width_mm=(1, 1, 1), verbose=False, kernel=None):
    if kernel is None:
        kernel = "gaussian"
    maximum_displacements_voxels = convert_mm_to_voxels_3d(maximum_displacements_mm, reference_image)
    peak_width_voxels = convert_mm_to_voxels_3d(peak_width_mm, reference_image)
    displacement_field_array = precalculated_generate_random_peaked_vector_field(reference_image.GetSize()[::-1],
                                                                   maximum_displacements_voxels,
                                                                   peak_count, peak_width_voxels, verbose=verbose, kernel=kernel)
    displacement_field = sitk.GetImageFromArray(displacement_field_array, isVector=True)
    # inverse_filter = sitk.InverseDisplacementFieldImageFilter()
    # inverse_filter.SetReferenceImage(vector_image)
    # displacement_field = inverse_filter.Execute(vector_image)
    displacement_field.CopyInformation(reference_image)
    transform = sitk.DisplacementFieldTransform(displacement_field)
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


def randomly_deform_image(image, maximum_displacements_mm=(10, 10, 10), peak_count=1, peak_width_mm=(1, 1, 1),
                          default_value=0, verbose=False):
    if type(maximum_displacements_mm) is int or type(maximum_displacements_mm) is float:
        maximum_displacements_mm = (maximum_displacements_mm, maximum_displacements_mm, maximum_displacements_mm)
    if type(peak_width_mm) is int or type(peak_width_mm) is float:
        peak_width_mm = (peak_width_mm, peak_width_mm, peak_width_mm)
    transform = generate_random_vector_field_transform(image, maximum_displacements_mm, peak_count, peak_width_mm, verbose=verbose)
    deformed_image = apply_transform(
        input_image=image,
        reference_image=image,
        transform=transform,
        default_value=default_value,
        interpolator=sitk.sitkLinear
    )
    return deformed_image, transform
