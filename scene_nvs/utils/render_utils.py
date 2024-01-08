import torch
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer.points import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.structures import Pointclouds

# from timings import log_time


class CustomRenderer(PointsRenderer):
    def __init__(self, cameras: PerspectiveCameras, im_size: tuple, device) -> None:
        self.device = device
        cameras = cameras.to(self.device)

        raster_settings = PointsRasterizationSettings(
            image_size=im_size, radius=0.01, points_per_pixel=1, bin_size=None
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        super().__init__(rasterizer=rasterizer, compositor=AlphaCompositor())

    def forward(self, point_clouds: Pointclouds) -> torch.Tensor:
        if point_clouds.points_padded().shape[1] > 100000:
            point_clouds = point_clouds.subsample(100000)

        point_clouds = point_clouds.to(self.device)

        fragments = self.rasterize(point_clouds)
        point_dists = fragments.zbuf.squeeze(-1)  # (b, h, w)

        # images = self.render(point_clouds)
        return point_dists

    # @log_time
    def rasterize(self, points: Pointclouds) -> torch.Tensor:
        fragments = self.rasterizer(points)
        return fragments

    # @log_time
    def render(self, points: Pointclouds) -> torch.Tensor:
        images = self(points)
        return images
