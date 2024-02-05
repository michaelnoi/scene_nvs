import torch
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer.points import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.structures import Pointclouds

# from utils.timings import log_time


class CustomRenderer(PointsRenderer):
    def __init__(
        self,
        cameras: PerspectiveCameras,
        im_size: tuple,
        device: str,
        radius: float = 0.01,
        points_per_pixel: int = 1,
    ) -> None:
        self.device = device
        cameras = cameras.to(self.device)

        raster_settings = PointsRasterizationSettings(
            image_size=im_size,
            radius=radius,
            points_per_pixel=points_per_pixel,
            bin_size=None,
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        super().__init__(rasterizer=rasterizer, compositor=AlphaCompositor())

    def render_depth_or_image(
        self, point_clouds: Pointclouds, only_zbuf: bool = False
    ) -> torch.Tensor:
        # if point_clouds.points_padded().shape[-2] > 100000:
        #     point_clouds = point_clouds.subsample(100000)

        point_clouds = point_clouds.to(self.device)

        if only_zbuf:
            fragments = self.rasterize(point_clouds)
            point_dists = fragments.zbuf.squeeze(-1)  # (b, h, w)
            return point_dists
        else:
            images = self.render(point_clouds)
            return images

    # @log_time
    def rasterize(self, points: Pointclouds) -> torch.Tensor:
        fragments = self.rasterizer(points)
        return fragments

    # @log_time
    def render(self, points: Pointclouds) -> torch.Tensor:
        images = self(points)
        return images
