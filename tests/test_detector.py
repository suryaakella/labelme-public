from labeling.detector import ObjectDetector


def test_detector_init():
    det = ObjectDetector()
    assert det.confidence == 0.3
    assert det.batch_size == 16
