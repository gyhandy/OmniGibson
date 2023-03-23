///////////////////////////////////////////////////////////////////////////////
// HTC Tracking System class
// Different coordinate systems:
//	OmniGibson: 		forward +y, right +x, up +z
//	SRanipal: 			forward +x, right +z, up +y
//	ViveHandTracking:	forward +z, right +x, up +z
///////////////////////////////////////////////////////////////////////////////


#pragma once

#include "SRanipal.h"
#include "SRanipal_Eye.h"
#include "SRanipal_Enums.h"
#include "interface_gesture.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <thread>

namespace py = pybind11;


class HTCTrackingSys {
public:
	// eye tracking variables
	bool useEyeTracking;
	bool shouldShutDownEyeTracking;
	std::thread* eyeTrackingThread;
	ViveSR::anipal::Eye::EyeData eyeData;

	// Struct storing eye data for SRanipal
	struct EyeTrackingData {
		bool isCombinedPoseValid = false;
		bool isLeftPoseValid = false;
		bool isRightPoseValid = false;
		// All in world space
		glm::vec3 combinedOrigin;
		glm::vec3 combinedDirection;
		glm::vec3 leftEyeOrigin;
		glm::vec3 leftEyeDirection;
		glm::vec3 rightEyeOrigin;
		glm::vec3 rightEyeDirection;
		// Both in mm
		float leftPupilDiameter;
		float rightPupilDiameter;
		// Both in 0~1
		float leftEyeOpenness;
		float rightEyeOpenness;
	};
	EyeTrackingData eyeTrackingData;
	
	// Vive Hand Tracking variables
	bool useHandTracking;
	const GestureResult* gestureResult;
	int lastGestureFrameIndex;
	float leftHandConfidence;
	float rightHandConfidence;
	py::list leftHandPosition;
	py::list rightHandPosition;
	py::list leftHandJointPosition;
	py::list rightHandJointPosition;
	py::list leftHandJointRotation;
	py::list rightHandJointRotation;

	HTCTrackingSys(bool useEyeTracking, bool useHandTracking);
	
	void start();

	void stop();
	
	py::list getHandTrackingData();

	py::list getEyeTrackingData();

	bool hasEyeTrackingSupport();


private:
	void pollAnipal();
};