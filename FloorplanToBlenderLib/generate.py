from . import IO
from . import const
from . import transform
import numpy as np

from FloorplanToBlenderLib.generator import Door, Floor, Room, Wall, Window, Table, Bed, Chair, Sofa, KitchenItem, Toilet, Bathtub

def generate_all_files(
    floorplan,
    info,
    world_direction=None,
    world_scale=np.array([1, 1, 1]),
    world_position=np.array([0, 0, 0]),
    world_rotation=np.array([0, 0, 0]),
):
    """
    Generate all data files
    @Param image path
    @Param dir build in negative or positive direction
    @Param info, boolean if should be printed
    @Param position, vector of float
    @Param rotation, vector of float
    @Return path to generated file, shape
    """
    if world_direction is None:
        world_direction = 1

    scale = [
        floorplan.scale[0] * world_scale[0],
        floorplan.scale[1] * world_scale[1],
        floorplan.scale[2] * world_scale[2],
    ]

    if info:
        print(
            " ----- Generate ",
            floorplan.image_path,
            " at pos ",
            transform.list_to_nparray(floorplan.position)
            + transform.list_to_nparray(world_position),
            " rot ",
            transform.list_to_nparray(floorplan.rotation)
            + transform.list_to_nparray(world_rotation),
            " scale ",
            scale,
            " -----",
        )

    # Get path to save data
    path = IO.create_new_floorplan_path(const.BASE_PATH)

    origin_path, shape = IO.find_reuseable_data(floorplan.image_path, const.BASE_PATH)

    if origin_path is None:
        origin_path = path

        _, gray, scale_factor = IO.read_image(floorplan.image_path, floorplan)

        if floorplan.floors:
            shape = Floor(gray, path, scale, info).shape

        if floorplan.walls:
            if shape is not None:
                new_shape = Wall(gray, path, scale, info).shape
                shape = validate_shape(shape, new_shape)
            else:
                shape = Wall(gray, path, scale, info).shape

        if floorplan.rooms:
            if shape is not None:
                new_shape = Room(gray, path, scale, info).shape
                shape = validate_shape(shape, new_shape)
            else:
                shape = Room(gray, path, scale, info).shape

        if floorplan.windows:
            Window(gray, path, floorplan.image_path, scale_factor, scale, info)

        if floorplan.doors:
            Door(gray, path, floorplan.image_path, scale_factor, scale, info)

        # Generate furniture with collision detection
        # First, detect all furniture types
        from FloorplanToBlenderLib import detect as furniture_detect

        furniture_raw = {}
        if hasattr(floorplan, 'tables') and floorplan.tables:
            furniture_raw['tables'] = furniture_detect.tables(floorplan.image_path, scale_factor)
        if hasattr(floorplan, 'beds') and floorplan.beds:
            furniture_raw['beds'] = furniture_detect.beds(floorplan.image_path, scale_factor)
        if hasattr(floorplan, 'chairs') and floorplan.chairs:
            furniture_raw['chairs'] = furniture_detect.chairs(floorplan.image_path, scale_factor)
        if hasattr(floorplan, 'sofas') and floorplan.sofas:
            furniture_raw['sofas'] = furniture_detect.sofas(floorplan.image_path, scale_factor)
        if hasattr(floorplan, 'kitchen') and floorplan.kitchen:
            furniture_raw['kitchen'] = furniture_detect.kitchen_items(floorplan.image_path, scale_factor)
        if hasattr(floorplan, 'toilets') and floorplan.toilets:
            furniture_raw['toilets'] = furniture_detect.toilets(floorplan.image_path, scale_factor)
        if hasattr(floorplan, 'bathtubs') and floorplan.bathtubs:
            furniture_raw['bathtubs'] = furniture_detect.bathtubs(floorplan.image_path, scale_factor)

        # Apply overlap removal filter
        furniture_filtered = furniture_detect.remove_furniture_overlaps(furniture_raw)

        # Now generate furniture using filtered data
        if 'tables' in furniture_filtered:
            Table(gray, path, floorplan.image_path, scale_factor, scale, info, furniture_filtered['tables'])

        if 'beds' in furniture_filtered:
            Bed(gray, path, floorplan.image_path, scale_factor, scale, info, furniture_filtered['beds'])

        if 'chairs' in furniture_filtered:
            Chair(gray, path, floorplan.image_path, scale_factor, scale, info, furniture_filtered['chairs'])

        if 'sofas' in furniture_filtered:
            Sofa(gray, path, floorplan.image_path, scale_factor, scale, info, furniture_filtered['sofas'])

        if 'kitchen' in furniture_filtered:
            KitchenItem(gray, path, floorplan.image_path, scale_factor, scale, info, furniture_filtered['kitchen'])

        if 'toilets' in furniture_filtered:
            Toilet(gray, path, floorplan.image_path, scale_factor, scale, info, furniture_filtered['toilets'])

        if 'bathtubs' in furniture_filtered:
            Bathtub(gray, path, floorplan.image_path, scale_factor, scale, info, furniture_filtered['bathtubs'])

    generate_transform_file(
        floorplan.image_path,
        path,
        info,
        floorplan.position,
        world_position,
        floorplan.rotation,
        world_rotation,
        scale,
        shape,
        path,
        origin_path,
    )

    if floorplan.position is not None:
        shape = [
            world_direction * shape[0] + floorplan.position[0] + world_position[0],
            world_direction * shape[1] + floorplan.position[1] + world_position[1],
            world_direction * shape[2] + floorplan.position[2] + world_position[2],
        ]

    if shape is None:
        shape = [0, 0, 0]

    return path, shape


def validate_shape(old_shape, new_shape):
    """
    Validate shape, use this to calculate a objects total shape
    @Param old_shape
    @Param new_shape
    @Return total shape
    """
    shape = [0, 0, 0]
    shape[0] = max(old_shape[0], new_shape[0])
    shape[1] = max(old_shape[1], new_shape[1])
    shape[2] = max(old_shape[2], new_shape[2])
    return shape


def generate_transform_file(
    img_path,
    path,
    info,
    position,
    world_position,
    rotation,
    world_rotation,
    scale,
    shape,
    data_path,
    origin_path,
):
    """
    Generate transform of file
    A transform contains information about an objects position, rotation.
    @Param img_path
    @Param info, boolean if should be printed
    @Param position, position vector
    @Param rotation, rotation vector
    @Param shape
    @Return transform
    """
    # create map
    transform = {}
    if position is None:
        transform[const.STR_POSITION] = np.array([0, 0, 0])
    else:
        transform[const.STR_POSITION] = position + world_position

    if scale is None:
        transform["scale"] = np.array([1, 1, 1])
    else:
        transform["scale"] = scale

    if rotation is None:
        transform[const.STR_ROTATION] = np.array([0, 0, 0])
    else:
        transform[const.STR_ROTATION] = rotation + world_rotation

    if shape is None:
        transform[const.STR_SHAPE] = np.array([0, 0, 0])
    else:
        transform[const.STR_SHAPE] = shape

    transform[const.STR_IMAGE_PATH] = img_path

    transform[const.STR_ORIGIN_PATH] = origin_path

    transform[const.STR_DATA_PATH] = data_path

    IO.save_to_file(path + "transform", transform, info)

    return transform
