import cv2 as cv
import cvzone
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv.VideoCapture(1)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
	while cap.isOpened():
		success, image = cap.read()
		if not success:
			print("Ignoring empty camera frame.")
			# If loading a video, use 'break' instead of 'continue'.
			continue
		image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
		results = face_mesh.process(image)
		# Draw the face mesh annotations on the image.
		image.flags.writeable = True
		image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
		if results.multi_face_landmarks:
			for face_landmarks in results.multi_face_landmarks:
				mp_drawing.draw_landmarks(
					image=image,
					landmark_list=face_landmarks,
					connections=mp_face_mesh.FACEMESH_TESSELATION,
					landmark_drawing_spec=None,
					connection_drawing_spec=mp_drawing_styles
					.get_default_face_mesh_tesselation_style())
				mp_drawing.draw_landmarks(
					image=image,
					landmark_list=face_landmarks,
					connections=mp_face_mesh.FACEMESH_CONTOURS,
					landmark_drawing_spec=None,
					connection_drawing_spec=mp_drawing_styles
					.get_default_face_mesh_contours_style())
				mp_drawing.draw_landmarks(
					image=image,
					landmark_list=face_landmarks,
					connections=mp_face_mesh.FACEMESH_IRISES,
					landmark_drawing_spec=None,
					connection_drawing_spec=mp_drawing_styles
					.get_default_face_mesh_iris_connections_style())
		# Flip the image horizontally for a selfie-view display.
		cv.imshow('Camera', image)
		if cv.waitKey(5) & 0xFF == 27:
			break
	cap.release()
	cv.destroyAllWindows()
