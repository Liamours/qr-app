import mediapipe as mp

class FaceMesh:
    def __init__(self, max_faces=2, refine=True, min_detect=0.5, min_track=0.5):
        self._mp = mp.solutions.face_mesh
        self.mesh = self._mp.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine,
            min_detection_confidence=min_detect,
            min_tracking_confidence=min_track,
        )

    def process(self, rgb_frame):
        return self.mesh.process(rgb_frame)

    def close(self):
        self.mesh.close()
