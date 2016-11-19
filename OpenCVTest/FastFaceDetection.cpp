#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <deque>
#include <fstream>

using namespace std;
using namespace cv;

// forward declarations
pair<Point, Point> eye_candidates(Mat& img);

void log_statistics(double stat) {
	return;
	static ofstream stream("stat.out");
	stream.precision(6);
	stream << fixed << stat << endl;
}
int main() {
	int camera_id = 0;
	VideoCapture webcam;
	Mat frame;

	if (!webcam.open(camera_id))
		return cerr << "Cannot capture from camera #" << camera_id << endl, -1;

	if (webcam.isOpened()) {
		cout << "Video capturing has been started ..." << endl;
		while (true) {
			webcam >> frame;
			if (frame.empty()) break;
			auto eyes = eye_candidates(frame);
			int c = waitKey(10);
			if (c == 27 || c == 'q' || c == 'Q')
				break;
		}
	}

	return 0;
}

namespace filter {
	void grayscale(Mat& img) {
		// converts image to grayscale
		cvtColor(img, img, COLOR_BGR2GRAY);
	}
	void scale(Mat& img, double factor) {
		// resize and scale image by a factor
		double fx = 1.0 / factor;
		Mat small;
		resize(img, small, Size(), fx, fx, INTER_LINEAR);
		img = small;
	}
	void equalize(Mat& img) {
		// equalize intensity histograms
		equalizeHist(img, img);
	}
	void median(Mat& img) {
		medianBlur(img, img, 3);
	}
	auto bgsub = createBackgroundSubtractorKNN();
	void subtract_background(Mat& img) {
		// receive the background 
		Mat mask;
		bgsub->apply(img, mask);
		// saturate black and white values
		Mat foreground;
		img.copyTo(foreground, mask);
		img = foreground;
	}
}

pair<Point, Point> eye_candidates(Mat& original_frame) {
	static CascadeClassifier face_cascade, eyes_cascade;
	if (face_cascade.empty()) {
		face_cascade.load("data/haarcascades/haarcascade_frontalface_alt.xml");
		eyes_cascade.load("data/haarcascades/haarcascade_eye.xml");
	}
	auto time_start = cvGetTickCount();
	Mat frame = original_frame.clone();
	Mat img = frame.clone();

	// apply filters
	filter::grayscale(img);
	filter::equalize(img);
	filter::median(img);

	// detect faces
	vector<Rect> faces;
	face_cascade.detectMultiScale(img, faces, 1.2, 3, CV_HAAR_FIND_BIGGEST_OBJECT, Size(80, 80));

	// find the biggest face
	int max_size = 0;
	Rect *max_face = NULL;
	for (auto& face : faces) {
		int size = face.width * face.height;
		if (size > max_size) {
			max_size = size;
			max_face = &face;
		}
	}

	static int lx, ly, rx, ry;
	if (max_face) {
		// Ajmera et. al approximation
		// approximate eye coordinates based on frontal face rectangle
		int x = max_face->x;
		int y = max_face->y;
		// rectangle(frame, *max_face, Scalar(255, 0, 0), 4);
		max_face->height *= 2.0 / 3.0;
		ry = ly = cvRound(y + max_face->height * 5 / 8.0);
		rx = cvRound(x + max_face->width * 5 / 16.0);
		lx = cvRound(x + max_face->width * 11 / 16.0);
	} else {
		// find the eyes
		vector<Rect> eyes;
		eyes_cascade.detectMultiScale(img, eyes, 1.75, 4, CV_HAAR_DO_CANNY_PRUNING, Size(40, 40), Size(80, 80));
		for (auto& eye : eyes)
			rectangle(frame, eye, Scalar(0, 0, 255));
		// get the eyes nearest to the interpolated points
		double best = HUGE_VAL;
		int LX, LY, RX, RY;
		double angle = atan2(ly - ry, lx - rx);
		for (int i = 0; i < eyes.size(); ++i) {
			double x1 = eyes[i].x + eyes[i].width / 2.0;
			double y1 = eyes[i].y + eyes[i].height / 2.0;
			for (int j = 0; j < eyes.size(); ++j) {
				if (i == j) continue;
				double x2 = eyes[j].x + eyes[j].width / 2.0;
				double y2 = eyes[j].y + eyes[j].height / 2.0;
				if (hypot(x2 - x1, y2 - y1) - eyes[i].width < 2)
					continue;
				double test = atan2(y2 - y1, x2 - x1);
				double dist = abs(test - angle);
				if (dist < best) {
					best = dist;
					LX = cvRound(x2);
					LY = cvRound(y2);
					RX = cvRound(x1);
					RY = cvRound(y1);
				}
			}
		}
		if (best != HUGE_VAL) {
			if (LX < RX) swap(LX, RX), swap(LY, RY);
			lx = LX;
			ly = LY;
			rx = RX;
			ry = RY;
		}
	}
	auto time_end = cvGetTickCount();
	double delta_time = (time_end - time_start) / (cvGetTickFrequency() * 1000.0);
	printf("time = %.4lfms\n", delta_time);
	circle(frame, {lx, ly}, 20, Scalar(255, 0, 0), 4);
	circle(frame, {rx, ry}, 20, Scalar(0, 255, 0), 4);
	imshow("Eye Detection", frame);
	return {{lx, ly}, {rx, ry}};
}

