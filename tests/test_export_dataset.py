from pathlib import Path
from zipfile import ZipFile
import json

import cv2
import numpy as np

from processing.export_dataset import DatasetFrame, export_r3d, export_tum


def _frame(tmp_path: Path) -> DatasetFrame:
    image = tmp_path / "source.jpg"; cv2.imwrite(str(image), np.zeros((4, 6, 3), np.uint8))
    pose = np.eye(4); pose[:3, 3] = [1, 2, 3]
    return DatasetFrame(1.25, image, np.full((4, 6), 2.0, np.float32), np.full((4, 6), 2, np.uint8), pose, np.array([[10,0,3],[0,11,2],[0,0,1]], float))


def test_export_tum_writes_millimetre_depth_and_pose_columns(tmp_path: Path):
    output = tmp_path / "tum"
    export_tum([_frame(tmp_path)], output)
    depth = cv2.imread(str(output / "depth/000000.png"), cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16 and depth[0, 0] == 2000
    assert (output / "color.txt").read_text().strip().endswith("rgb/000000.jpg")
    values = (output / "groundtruth.txt").read_text().strip().split()
    assert len(values) == 8 and values[1:4] == ["1.000000000", "2.000000000", "3.000000000"]
    assert "d400_color_optical_frame" in (output / "sensors.yaml").read_text()
    sensors = cv2.FileStorage(str(output / "sensors.yaml"), cv2.FILE_STORAGE_READ)
    np.testing.assert_allclose(sensors.getNode("d400_color_optical_frame").getNode("intrinsics").mat(), [[10, 11, 3, 2]])
    sensors.release()
    transforms = cv2.FileStorage(str(output / "trans_matrix.yaml"), cv2.FILE_STORAGE_READ)
    assert transforms.getNode("trans_matrix").size() == 2
    transforms.release()


def test_export_r3d_writes_cross_archive_contract(tmp_path: Path):
    archive = tmp_path / "sample.r3d"
    export_r3d([_frame(tmp_path)], archive, compressor=lambda data: b"z" + data)
    with ZipFile(archive) as zipped:
        assert {"metadata", "rgbd/0.jpg", "rgbd/0.depth", "rgbd/0.conf"} <= set(zipped.namelist())
        metadata = json.loads(zipped.read("metadata"))
        assert set(["w", "h", "dw", "dh", "fps", "K", "poses", "initPose"]) <= set(metadata)
        assert metadata["poses"][0][4:] == [1.0, 2.0, 3.0]
        assert zipped.read("rgbd/0.depth").startswith(b"z")
