"""ProcessingTechniques – LiDAR post-processing algorithms.

Submodules
----------
geo        – CRS detection, coordinate reprojection
filters    – Point-cloud filters (SOR, voxel, elevation, ground seg)
satellite  – Satellite imagery fetching & point-cloud colorization
"""

from .geo import (
    extract_crs_wkt,
    extract_epsg_from_geokeys,
    build_crs,
    transform_to_wgs84,
    describe_coord_ranges,
)

from .filters import (
    statistical_outlier_removal,
    voxel_downsample,
    elevation_clip,
    ground_segmentation,
)

from .satellite import (
    fetch_satellite_tiles,
    colorize_points_from_satellite,
    generate_satellite_colors,
)
